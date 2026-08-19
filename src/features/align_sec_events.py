from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal


# ============================================================
# ŚCIEŻKI
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sentiment_filing_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sec_event_features.csv"
)


# ============================================================
# KONFIGURACJA
# ============================================================

MARKET_CALENDAR_NAME = "NASDAQ"

MARKET_TIMEZONE = "America/New_York"


# ============================================================
# WALIDACJA
# ============================================================

def validate_dataframe(
    df: pd.DataFrame,
) -> None:

    required_columns = [
        "Ticker",
        "Filing_Date",
        "Acceptance_DateTime_ET",
        "Accession",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Brakuje wymaganych kolumn w "
            "sentiment_filing_features.csv:\n"
            + "\n".join(missing_columns)
        )

    if df["Acceptance_DateTime_ET"].isna().any():
        raise ValueError(
            "Znaleziono filing bez "
            "Acceptance_DateTime_ET."
        )


# ============================================================
# PRZYGOTOWANIE KALENDARZA
# ============================================================

def create_market_schedule(
    df: pd.DataFrame,
) -> pd.DataFrame:

    acceptance_times = pd.to_datetime(
        df["Acceptance_DateTime_ET"],
        format="mixed",
        utc=True,
    )

    min_date = (
        acceptance_times
        .min()
        .date()
    )

    max_date = (
        acceptance_times
        .max()
        .date()
    )

    # Zapas potrzebny do wyszukiwania
    # poprzednich i następnych sesji.
    schedule_start = (
        pd.Timestamp(min_date)
        - pd.Timedelta(days=30)
    )

    schedule_end = (
        pd.Timestamp(max_date)
        + pd.Timedelta(days=30)
    )

    calendar = mcal.get_calendar(
        MARKET_CALENDAR_NAME
    )

    schedule = calendar.schedule(
        start_date=schedule_start,
        end_date=schedule_end,
    )

    if schedule.empty:
        raise ValueError(
            "Kalendarz giełdowy jest pusty."
        )

    return schedule


# ============================================================
# POMOCNICZE FUNKCJE SESJI
# ============================================================

def get_session_dates(
    schedule: pd.DataFrame,
) -> pd.DatetimeIndex:

    return pd.DatetimeIndex(
        schedule.index
    ).normalize()


def get_next_session(
    session_dates: pd.DatetimeIndex,
    date: pd.Timestamp,
) -> pd.Timestamp:

    normalized_date = date.normalize()

    future_sessions = session_dates[
        session_dates > normalized_date
    ]

    if len(future_sessions) == 0:
        raise ValueError(
            f"Nie znaleziono następnej sesji "
            f"dla {date}."
        )

    return future_sessions[0]


def get_previous_session(
    session_dates: pd.DatetimeIndex,
    date: pd.Timestamp,
) -> pd.Timestamp:

    normalized_date = date.normalize()

    previous_sessions = session_dates[
        session_dates < normalized_date
    ]

    if len(previous_sessions) == 0:
        raise ValueError(
            f"Nie znaleziono poprzedniej sesji "
            f"dla {date}."
        )

    return previous_sessions[-1]


# ============================================================
# ALIGNMENT JEDNEGO FILINGU
# ============================================================

def align_single_event(
    acceptance_datetime: pd.Timestamp,
    schedule: pd.DataFrame,
    session_dates: pd.DatetimeIndex,
) -> dict:

    # ========================================================
    # ACCEPTANCE DATETIME -> UTC / ET
    # ========================================================

    acceptance_utc = pd.to_datetime(
        acceptance_datetime,
        utc=True,
    )

    acceptance_et = (
        acceptance_utc
        .tz_convert(
            MARKET_TIMEZONE
        )
    )

    filing_date = pd.Timestamp(
        acceptance_et.date()
    )

    # ========================================================
    # CZY DZIEŃ PUBLIKACJI JEST DNIEM SESYJNYM?
    # ========================================================

    is_trading_day = (
        filing_date
        in session_dates
    )

    # ========================================================
    # WEEKEND / ŚWIĘTO
    # ========================================================

    if not is_trading_day:

        event_session = get_next_session(
            session_dates=session_dates,
            date=filing_date,
        )

        # Ostatnia pełna sesja sprzed publikacji.
        feature_cutoff_session = (
            get_previous_session(
                session_dates=session_dates,
                date=event_session,
            )
        )

        return {
            "Publication_Period":
                "NON_TRADING_DAY",

            "Filing_Session_Open_ET":
                pd.NaT,

            "Filing_Session_Close_ET":
                pd.NaT,

            "Feature_Cutoff_Session":
                feature_cutoff_session.date(),

            "Event_Session":
                event_session.date(),
        }

    # ========================================================
    # GODZINY SESJI DNIA PUBLIKACJI
    # ========================================================

    schedule_row = schedule.loc[
        filing_date
    ]

    market_open_utc = pd.Timestamp(
        schedule_row[
            "market_open"
        ]
    )

    market_close_utc = pd.Timestamp(
        schedule_row[
            "market_close"
        ]
    )

    market_open_et = (
        market_open_utc
        .tz_convert(
            MARKET_TIMEZONE
        )
    )

    market_close_et = (
        market_close_utc
        .tz_convert(
            MARKET_TIMEZONE
        )
    )

    # ========================================================
    # PRE-MARKET
    # ========================================================

    if acceptance_utc < market_open_utc:

        publication_period = (
            "PRE_MARKET"
        )

        # Rynek jeszcze się nie otworzył,
        # więc reakcja może rozpocząć się
        # podczas tej samej sesji.
        event_session = filing_date

        # Features mogą korzystać najwyżej
        # z poprzedniej zakończonej sesji.
        feature_cutoff_session = (
            get_previous_session(
                session_dates=session_dates,
                date=filing_date,
            )
        )

    # ========================================================
    # AFTER-HOURS
    # ========================================================

    elif acceptance_utc >= market_close_utc:

        publication_period = (
            "AFTER_HOURS"
        )

        # Rynek zareaguje podczas kolejnej sesji.
        event_session = get_next_session(
            session_dates=session_dates,
            date=filing_date,
        )

        # Cała sesja filing_date zakończyła się
        # jeszcze przed publikacją raportu.
        feature_cutoff_session = (
            filing_date
        )

    # ========================================================
    # INTRADAY
    # ========================================================

    else:

        publication_period = (
            "INTRADAY"
        )

        # Przy danych dziennych przyjmujemy
        # konserwatywnie następną pełną sesję.
        event_session = get_next_session(
            session_dates=session_dates,
            date=filing_date,
        )

        # Nie możemy użyć Close z dnia publikacji,
        # ponieważ filing pojawił się już podczas sesji.
        feature_cutoff_session = (
            get_previous_session(
                session_dates=session_dates,
                date=filing_date,
            )
        )

    return {
        "Publication_Period":
            publication_period,

        "Filing_Session_Open_ET":
            market_open_et.isoformat(),

        "Filing_Session_Close_ET":
            market_close_et.isoformat(),

        "Feature_Cutoff_Session":
            feature_cutoff_session.date(),

        "Event_Session":
            event_session.date(),
    }


# ============================================================
# ALIGNMENT CAŁEGO DATASETU
# ============================================================

def align_sec_events(
    df: pd.DataFrame,
) -> pd.DataFrame:

    validate_dataframe(
        df
    )

    schedule = create_market_schedule(
        df
    )

    session_dates = get_session_dates(
        schedule
    )

    aligned_rows = []

    for _, row in df.iterrows():

        alignment = align_single_event(
            acceptance_datetime=row[
                "Acceptance_DateTime_ET"
            ],
            schedule=schedule,
            session_dates=session_dates,
        )

        result_row = row.to_dict()

        result_row.update(
            alignment
        )

        aligned_rows.append(
            result_row
        )

    aligned_df = pd.DataFrame(
        aligned_rows
    )

    # ========================================================
    # FORMAT DAT
    # ========================================================

    aligned_df[
        "Filing_Date"
    ] = pd.to_datetime(
        aligned_df[
            "Filing_Date"
        ]
    ).dt.date

    aligned_df[
        "Feature_Cutoff_Session"
    ] = pd.to_datetime(
        aligned_df[
            "Feature_Cutoff_Session"
        ]
    ).dt.date

    aligned_df[
        "Event_Session"
    ] = pd.to_datetime(
        aligned_df[
            "Event_Session"
        ]
    ).dt.date

    # ========================================================
    # WALIDACJA LOGIKI CZASOWEJ
    # ========================================================

    invalid_cutoff = (
        pd.to_datetime(
            aligned_df[
                "Feature_Cutoff_Session"
            ]
        )
        >=
        pd.to_datetime(
            aligned_df[
                "Event_Session"
            ]
        )
    )

    if invalid_cutoff.any():
        invalid_rows = aligned_df.loc[
            invalid_cutoff,
            [
                "Ticker",
                "Accession",
                "Publication_Period",
                "Feature_Cutoff_Session",
                "Event_Session",
            ],
        ]

        raise ValueError(
            "Feature_Cutoff_Session musi być "
            "wcześniejsza niż Event_Session.\n"
            f"{invalid_rows}"
        )

    # ========================================================
    # SORTOWANIE
    # ========================================================

    aligned_df = (
        aligned_df.sort_values(
            by=[
                "Ticker",
                "Filing_Date",
                "Accession",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return aligned_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 80
    )

    print(
        "SYNCHRONIZACJA SEC Z SESJAMI GIEŁDOWYMI"
    )

    print(
        "=" * 80
    )

    print(
        f"\nPlik wejściowy:\n"
        f"{INPUT_FILE}"
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku:\n"
            f"{INPUT_FILE}"
        )

    # ========================================================
    # WCZYTANIE
    # ========================================================

    df_filings = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"\nLiczba filingów: "
        f"{len(df_filings)}"
    )

    # ========================================================
    # ALIGNMENT
    # ========================================================

    df_aligned = align_sec_events(
        df_filings
    )

    # ========================================================
    # ZAPIS
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df_aligned.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # PODGLĄD
    # ========================================================

    columns_to_show = [
        "Ticker",
        "Filing_Date",
        "Acceptance_DateTime_ET",
        "Publication_Period",
        "Filing_Session_Open_ET",
        "Filing_Session_Close_ET",
        "Feature_Cutoff_Session",
        "Event_Session",
        "Accession",
    ]

    print(
        "\n"
        + "=" * 80
    )

    print(
        "WYNIK ALIGNMENTU"
    )

    print(
        "=" * 80
    )

    print(
        df_aligned[
            columns_to_show
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # ROZKŁAD TYPÓW PUBLIKACJI
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "ROZKŁAD PUBLICATION PERIOD"
    )

    print(
        "=" * 80
    )

    print(
        df_aligned[
            "Publication_Period"
        ].value_counts(
            dropna=False
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"Wyniki zapisano do:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        "=" * 80
    )