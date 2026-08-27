##########################################################
# Ten plik odpowiada za to aby poprawnie dopasować daty
# Usstalenie momentu publikacji względem sesji
########################################################


from pathlib import Path
import logging

import pandas as pd
import pandas_market_calendars as mcal

logger = logging.getLogger(__name__)

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

# Jaki kalendarz i giełda
MARKET_CALENDAR_NAME = "NASDAQ"
MARKET_TIMEZONE = "America/New_York"


# Walidacja
def validate_dataframe(df: pd.DataFrame) -> None:

    if df.empty:
        raise ValueError(
            "Pusty DataFrame"
        )

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
            "sentiment_filing_features.csv: "
            f"{missing_columns}"
        )

    missing_timestamp = df["Acceptance_DateTime_ET"].isna()

    if missing_timestamp.any():
        invalid_filings = df.loc[
            missing_timestamp,
            [
                "Ticker",
                "Filing_Date",
                "Accession",
            ],
        ]

        raise ValueError(
            "Znaleziono filing bez "
            "Acceptance_DateTime_ET:\n"
            f"{invalid_filings.to_string(index=False)}"
        )


# Kalendarz
def create_market_schedule(df: pd.DataFrame,) -> pd.DataFrame:

    # Zamiana na timestamp
    acceptance_times = (
        pd.to_datetime(
            df["Acceptance_DateTime_ET"],
            format="mixed",
            utc=True,
        )
        .dt.tz_convert(MARKET_TIMEZONE)
    )

    min_date = acceptance_times.min().date()
    

    max_date = acceptance_times.max().date()
    

    # Zapas potrzebny do wyszukiwania poprzednich i następnych sesji
    schedule_start = pd.Timestamp(min_date) - pd.Timedelta(days=30)

    schedule_end = pd.Timestamp(max_date) + pd.Timedelta(days=30)

    calendar = mcal.get_calendar(MARKET_CALENDAR_NAME)

    schedule = calendar.schedule(
        start_date=schedule_start,
        end_date=schedule_end,
    )

    if schedule.empty:
        raise ValueError("Kalendarz giełodwy jest pusty")

    return schedule


# Funkcje pomocnicze
def get_session_dates(schedule: pd.DataFrame) -> pd.DatetimeIndex:
    """Zamienia indeks kalendarza n aliste dat sesyjnch"""
    return pd.DatetimeIndex(schedule.index).normalize()


def get_next_session(
    session_dates: pd.DatetimeIndex,
    date: pd.Timestamp,
) -> pd.Timestamp:
    """
    Jaka jest pierwsza sesja po danej dacie
    """

    normalized_date = date.normalize()

    future_sessions = session_dates[session_dates > normalized_date]

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
    """
    Jaka jest ostatnia sesja przed przyszłą sesja
    """

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


# Dla jednego filingu
def align_single_event(
    acceptance_datetime: pd.Timestamp,
    schedule: pd.DataFrame,
    session_dates: pd.DatetimeIndex,
) -> dict:
    """
    Przypisuje filing SEC do odpowiedniej sesji rynkowej i wyznacza ostatnią sesję
    dostępną do budowy cech
    """

    acceptance_timestamp = pd.Timestamp(acceptance_datetime)

    if acceptance_timestamp.tzinfo is None:
        raise ValueError(
            "Acceptance_DateTime_ET nie posiada "
            "informacji o strefie czasowej."
        )

    acceptance_utc = acceptance_timestamp.tz_convert("UTC")

    acceptance_et = acceptance_timestamp.tz_convert(MARKET_TIMEZONE)

    publication_date = pd.Timestamp(acceptance_et.date())

    is_trading_day = publication_date in session_dates

    market_open_et = pd.NaT
    market_close_et = pd.NaT

    # NON-TRADING DAY

    if not is_trading_day:

        publication_period = "NON_TRADING_DAY"

        event_session = get_next_session(
            session_dates=session_dates,
            date=publication_date,
        )

        feature_cutoff_session = get_previous_session(
                session_dates=session_dates,
                date=event_session,
            )


    # TRADING DAY
    else:

        schedule_row = schedule.loc[publication_date]

        market_open_utc = pd.Timestamp(schedule_row["market_open"])

        market_close_utc = pd.Timestamp(schedule_row["market_close"])

        market_open_et = market_open_utc.tz_convert(MARKET_TIMEZONE)

        market_close_et = market_close_utc.tz_convert(MARKET_TIMEZONE)
    

        # PRE-MARKET
        if acceptance_utc < market_open_utc:

            publication_period = "PRE_MARKET"

            event_session = publication_date

            feature_cutoff_session = get_previous_session(
                    session_dates=session_dates,
                    date=publication_date,
                )
    

        # AFTER-HOURS
        elif acceptance_utc >= market_close_utc:

            publication_period = "AFTER_HOURS"

            event_session = get_next_session(
                    session_dates=session_dates,
                    date=publication_date,
                )

            feature_cutoff_session = publication_date
            

        # INTRADAY
        else:

            publication_period = "INTRADAY"

            event_session = get_next_session(
                    session_dates=session_dates,
                    date=publication_date,
                )
            

            feature_cutoff_session = get_previous_session(
                    session_dates=session_dates,
                    date=publication_date,
                )
            

    # Walidacja przed data-leakage
    if feature_cutoff_session>= event_session:
        raise ValueError(
            "Feature_Cutoff_Session musi być "
            "wcześniejsza niż Event_Session."
        )

    return {
        "Publication_Period": publication_period,

        "Filing_Session_Open_ET": market_open_et,

        "Filing_Session_Close_ET": market_close_et,

        "Feature_Cutoff_Session": feature_cutoff_session.date(),

        "Event_Session": event_session.date(),
    }

# Alignment całego datasetu
def align_sec_events(df: pd.DataFrame) -> pd.DataFrame:

    validate_dataframe(df)

    schedule = create_market_schedule(df)

    session_dates = get_session_dates(schedule)

    aligned_rows = []

    for _, row in df.iterrows():

        alignment = align_single_event(
            acceptance_datetime=row["Acceptance_DateTime_ET"],
            schedule=schedule,
            session_dates=session_dates,
        )

        result_row = row.to_dict()

        result_row.update(alignment)

        aligned_rows.append(result_row)

    aligned_df = pd.DataFrame(aligned_rows)


    # Format dla dat
    aligned_df["Filing_Date"] = pd.to_datetime(
        aligned_df["Filing_Date"],
        errors="raise"
        ).dt.date

    aligned_df["Feature_Cutoff_Session"] = pd.to_datetime(
        aligned_df["Feature_Cutoff_Session"]
    ).dt.date

    aligned_df["Event_Session"] = pd.to_datetime(
        aligned_df["Event_Session"]
    ).dt.date

    # Walidacja dla logiki czasowej (niby juz była dal pojednynczych ale dla bezpieczenstwa)
    invalid_cutoff = (
        pd.to_datetime(aligned_df["Feature_Cutoff_Session"])
        >=
        pd.to_datetime(aligned_df["Event_Session"])
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
            f"{invalid_rows.to_string(index=False)}"
        )

    # Sortowanie
    aligned_df = aligned_df.sort_values(
            by=[
                "Ticker",
                "Filing_Date",
                "Accession",
            ]
        ).reset_index(drop=True)

    return aligned_df




def main() -> None:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {INPUT_FILE}")

    # Wczytanie danych
    df_filings = pd.read_csv(INPUT_FILE)

    logger.info("Liczba filingów wejściowych: %d", len(df_filings))

    # Alignment
    df_aligned = align_sec_events(df_filings)

    # Walidacja liczby wierszy
    if len(df_aligned) != len(df_filings):
        raise ValueError(
            "Alignment zmienił liczbę filingów: "
            f"wejście={len(df_filings)}, "
            f"wyjście={len(df_aligned)}."
        )

    # typy publikacji
    publication_period_counts = df_aligned["Publication_Period"].value_counts(dropna=False)
    

    logger.info("Rozkład Publication_Period:\n%s", publication_period_counts.to_string(),)

    # Zapis
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df_aligned.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    logger.info(
        "Zapisano %d filingów do %s",
        len(df_aligned),
        OUTPUT_FILE,
    )

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - "
            "%(levelname)s - "
            "%(name)s - "
            "%(message)s"
        ),
    )

    main()