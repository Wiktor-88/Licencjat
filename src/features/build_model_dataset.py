##############################################################################
# Ten plik odpowiada za budowe głównej ramki danych, ktróej
# będą używaly już modele ML
###############################################################################


from pathlib import Path
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEC_INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "sec_event_features.csv"
MARKET_INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "market_features.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "model_dataset.csv"


# Cechy rynkowe 
MARKET_FEATURE_COLUMNS = [
    "Log_Return_1D",
    "Log_Return_3D",
    "Log_Return_5D",
    "Volatility_14D",
    "Relative_Volume_20D",
    "RSI_14",
    "Price_to_SMA20",
    "Intraday_Return",
    "Daily_Range",
]

# Cechy Rolling Z-score

ROLLING_Z_FEATURE_COLUMNS = [
    "Log_Return_1D_Z60",
    "Log_Return_3D_Z60",
    "Log_Return_5D_Z60",
    "Volatility_14D_Z60",
    "Relative_Volume_20D_Z60",
    "Price_to_SMA20_Z60",
    "Intraday_Return_Z60",
    "Daily_Range_Z60",
]

BENCHMARK_TICKER = "QQQ"

BENCHMARK_FEATURE_COLUMNS = [
    "Log_Return_1D",
    "Log_Return_3D",
    "Log_Return_5D",
    "Volatility_14D",
]


# DODANIE CECH BENCHMARKU Z FEATURE CUTOFF SESSION
def merge_benchmark_cutoff_features(
    df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Dla każdego filingu zajduje stan benchmarku QQQ
    """
    
    # Wybór QQQ
    benchmark_df = market_df.loc[
        market_df["Ticker"] == BENCHMARK_TICKER,
        ["Date", "Adj_Close"] + BENCHMARK_FEATURE_COLUMNS,
    ].copy()

    if benchmark_df.empty:
        raise ValueError(
            f"Nie znaleziono benchmarku {BENCHMARK_TICKER} "
            "w market_features.csv."
        )

    benchmark_df = benchmark_df.rename(columns={
            "Date": "Feature_Cutoff_Session",
            "Adj_Close": "QQQ_Cutoff_Adj_Close",
            "Log_Return_1D": "QQQ_Log_Return_1D",
            "Log_Return_3D": "QQQ_Log_Return_3D",
            "Log_Return_5D": "QQQ_Log_Return_5D",
            "Volatility_14D": "QQQ_Volatility_14D",
        })

    result = df.merge(
        benchmark_df,
        on="Feature_Cutoff_Session",
        how="left",
        validate="many_to_one",
    )

    # Sprawdzenie czy QQQ dopassowało sie do każdego filingu
    missing_benchmark = result["QQQ_Cutoff_Adj_Close"].isna()

    if missing_benchmark.any():
        missing_rows = result.loc[
            missing_benchmark,
            ["Ticker", "Accession", "Feature_Cutoff_Session"],
        ]

        raise ValueError(
            "Brak danych QQQ dla części Feature_Cutoff_Session:\n"
            f"{missing_rows.to_string(index=False)}"
        )

    return result


# Walidacja danych SEC
def validate_sec_data(df: pd.DataFrame,) -> None:

    required_columns = [
        "Ticker",
        "Filing_Date",
        "Acceptance_DateTime_ET",
        "Accession",
        "Publication_Period",
        "Feature_Cutoff_Session",
        "Event_Session",
        "Mean_Net_Sentiment",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Brakuje wymaganych kolumn w "
            f"sec_event_features.csv: {missing_columns}"
        )

    # Każdy accession powinien występować dokładnie raz
    duplicate_mask = df.duplicated(
        subset=["Ticker", "Accession"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = df.loc[
            duplicate_mask,
            ["Ticker", "Accession"],
        ]

        raise ValueError(
            "Znaleziono zduplikowane filingi:\n"
            f"{duplicates.to_string(index=False)}"
        )

    # Kontrola czy nie ma nanów dla kluczowych wartości
    critical_columns = [
        "Ticker",
        "Accession",
        "Publication_Period",
        "Feature_Cutoff_Session",
        "Event_Session",
    ]

    missing_critical = df[critical_columns].isna().any(axis=1)

    if missing_critical.any():
        invalid_rows = df.loc[
            missing_critical,
            ["Ticker", "Accession"] + critical_columns[2:],
        ]

        raise ValueError(
            "Znaleziono filing z brakującymi danymi "
            "wymaganymi do budowy datasetu:\n"
            f"{invalid_rows.to_string(index=False)}"
        )


    valid_periods = {
        "PRE_MARKET",
        "INTRADAY",
        "AFTER_HOURS",
        "NON_TRADING_DAY",
    }

    unexpected_periods = set(df["Publication_Period"].unique()) - valid_periods

    if unexpected_periods:
        raise ValueError(
            "Niepoprawne wartości Publication_Period: "
            f"{unexpected_periods}"
        )

    cutoff_dates = pd.to_datetime(df["Feature_Cutoff_Session"], errors="raise")
    event_dates = pd.to_datetime(df["Event_Session"], errors="raise")

    if (cutoff_dates >= event_dates).any():
        raise ValueError(
            "Feature_Cutoff_Session musi być wcześniejsza "
            "niż Event_Session."
        )



# WALIDACJA DANYCH RYNKOWYCH
def validate_market_data(df: pd.DataFrame,) -> None:

    required_columns = [
        "Ticker",
        "Date",
        "Adj_Close",
    ] + MARKET_FEATURE_COLUMNS + ROLLING_Z_FEATURE_COLUMNS

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Brakuje wymaganych kolumn "
            "w market_features.csv:\n"
            + f"{missing_columns})"
        )

    duplicate_mask = df.duplicated(
        subset=["Ticker", "Date"],
        keep=False,
    )

    if duplicate_mask.any():

        duplicates = df.loc[
            duplicate_mask,
            ["Ticker", "Date"]
        ]

        raise ValueError(
            "Znaleziono zduplikowane "
            "Ticker + Date w market_features.csv:\n"
            f"{duplicates}"
        )


    adj_close = pd.to_numeric(
        df["Adj_Close"],
        errors="coerce",
    )

    # Zła cena <0, Nan lub inf , -inf
    invalid_price_mask = (
        adj_close.isna()
        | ~np.isfinite(adj_close)
        | (adj_close <= 0)
    )

    if invalid_price_mask.any():
        invalid_rows = df.loc[
            invalid_price_mask,
            ["Ticker", "Date", "Adj_Close"],
        ]

        raise ValueError(
            "Znaleziono niepoprawne wartości Adj_Close:\n"
            f"{invalid_rows.to_string(index=False)}"
        )


def parse_acceptance_utc(series: pd.Series) -> pd.Series:
    """
    Parsuje Acceptance_DateTime_ET do UTC

    Wymagamy jawnej informacji o strefie czasowej np.:
    2026-05-20T16:21:19-04:00

    Nie bierzemy naive datetime bo nie chcemy zgadywać strefy czasowej
    """

    timestamp_text = series.astype("string").str.strip()

    has_timezone = timestamp_text.str.contains(
        r"(?:Z|[+-]\d{2}:\d{2})$",
        regex=True,
        na=False,
    )

    if not has_timezone.all():
        invalid_values = timestamp_text[~has_timezone]

        raise ValueError(
            "Znaleziono brakujące Acceptance_DateTime_ET "
            "lub wartości bez jawnego offsetu strefy czasowej:\n"
            f"{invalid_values.to_string()}"
        )

    return pd.to_datetime(
        timestamp_text,
        format="mixed",
        utc=True,
        errors="raise",
    )



########## WALIDACJA MARKET MATCHES ###############
def validate_market_matches(df: pd.DataFrame) -> None:
    required_price_columns = [
        "Cutoff_Adj_Close",
        "Event_Adj_Close",
        "QQQ_Cutoff_Adj_Close",
        "QQQ_Event_Adj_Close",
    ]

    missing_columns = [
        column for column in required_price_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Brakuje kolumn wymaganych do walidacji "
            f"danych rynkowych: {missing_columns}"
        )

    price_data = df[required_price_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    invalid_price_mask = (
        price_data.isna()
        | ~np.isfinite(price_data)
        | (price_data <= 0)
    ).any(axis=1)

    if invalid_price_mask.any():
        invalid_rows = df.loc[
            invalid_price_mask,
            [
                "Ticker",
                "Accession",
                "Feature_Cutoff_Session",
                "Event_Session",
            ] + required_price_columns,
        ]

        raise ValueError(
            "Znaleziono brakujące lub niepoprawne "
            "dane cenowe spółki lub QQQ:\n"
            f"{invalid_rows.to_string(index=False)}"
        )



# Momentum dla sentymentów
def add_sentiment_momentum(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Liczy różnicę między sentymentem bieżącego filingu a średnim sentymentem
    maksymalnie 3 wcześniejszych filingów z dostępnym sentymentem

    Finingi bez Mean_Net_Sentiment nie są traktowane jako neutralne i nie wchodzą
    do historii sentymentu
    """


    df = df.copy()

    # pomocnicza kolumna - pozniej jest usuwana
    df["_Acceptance_UTC"] = parse_acceptance_utc(df["Acceptance_DateTime_ET"])

    # Sprowadznie na ten sam czas
    same_time_mask = df.duplicated(
        subset=["Ticker", "_Acceptance_UTC"],
        keep=False,
    )

    if same_time_mask.any():
        same_time_rows = df.loc[
            same_time_mask,
            ["Ticker", "Accession", "_Acceptance_UTC"],
        ]

        raise ValueError(
            "Znaleziono kilka filingów tego samego tickera "
            "z identycznym Acceptance_DateTime_ET:\n"
            f"{same_time_rows.to_string(index=False)}"
        )



    df.sort_values(
        by=["Ticker", "_Acceptance_UTC", "Accession"],
        inplace=True,
        ignore_index=True,
    )

    df["Previous_Sentiment_Mean_3"] = np.nan

    df["Sentiment_History_Count_3"] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Int64",
    )


    # TYLKO FILINGI Z DOSTĘPNYM SENTYMENTEM
    valid_mask = df["Mean_Net_Sentiment"].notna()

    sentiment_history = df.loc[
        valid_mask,
        ["Ticker", "Mean_Net_Sentiment"]
    ].copy()

   
    # Poprzedni sentyment - shift(1) nie bierze obecnego tylko poprzednie
    shifted_sentiment = (
        sentiment_history
        .groupby("Ticker", sort=False)["Mean_Net_Sentiment"].shift(1)
    )

    # srednia z poprzenich
    previous_mean = (
        shifted_sentiment
        .groupby(sentiment_history["Ticker"], sort=False)
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # zliczanie
    previous_count = (
        shifted_sentiment
        .groupby(sentiment_history["Ticker"], sort=False)
        .rolling(window=3, min_periods=1)
        .count()
        .reset_index(level=0, drop=True)
        .astype("Int64")
    )

    df.loc[
        sentiment_history.index,
        "Previous_Sentiment_Mean_3",
    ] = previous_mean

    df.loc[
        sentiment_history.index,
        "Sentiment_History_Count_3",
    ] = previous_count


    ##### Momentum ####
    df["Sentiment_Momentum_3"] = df["Mean_Net_Sentiment"] - df["Previous_Sentiment_Mean_3"]
    

    df.drop(
        columns=["_Acceptance_UTC"],
        inplace=True,
    )

    return df



# ŁĄCZENIE CECH RYNKOWYCH Z FEATURE CUTOFF SESSION
def merge_cutoff_market_features(
    sec_df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> pd.DataFrame:

    cutoff_market = market_df[
        ["Ticker", "Date", "Adj_Close"]
        + MARKET_FEATURE_COLUMNS
        + ROLLING_Z_FEATURE_COLUMNS
    ].copy()

    cutoff_market = cutoff_market.rename(columns={
            "Date": "Feature_Cutoff_Session",
            "Adj_Close": "Cutoff_Adj_Close",
        })

    result = sec_df.merge(
        cutoff_market,
        on=["Ticker", "Feature_Cutoff_Session"],
        how="left",
        validate="many_to_one",
    )

    return result



# DODANIE CENY Z EVENT SESSION
def merge_event_price(
    df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> pd.DataFrame:

    event_prices = market_df[
        ["Ticker", "Date", "Adj_Close"]
    ].copy()

    event_prices = event_prices.rename(columns={
        "Date": "Event_Session",
        "Adj_Close": "Event_Adj_Close",
    })

    result = df.merge(
        event_prices,
        on=["Ticker", "Event_Session"],
        how="left",
        validate="many_to_one",
    )

    return result



# DODANIE CENY BENCHMARKU Z EVENT SESSION
def merge_benchmark_event_price(
    df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    benchmark_prices = market_df.loc[
        market_df["Ticker"] == BENCHMARK_TICKER,
        ["Date", "Adj_Close"]
    ].copy()

    if benchmark_prices.empty:
        raise ValueError(
            f"Nie znaleziono benchmarku {BENCHMARK_TICKER} "
            "w market_features.csv."
        )

    benchmark_prices = benchmark_prices.rename(columns={
            "Date": "Event_Session",
            "Adj_Close": "QQQ_Event_Adj_Close",
        })

    result = df.merge(
        benchmark_prices,
        on="Event_Session",
        how="left",
        validate="many_to_one",
    )

    return result


# RELATYWNE CECHY SPÓŁKA VS QQQ
def add_relative_market_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["Stock_vs_QQQ_1D"] = df["Log_Return_1D"]- df["QQQ_Log_Return_1D"]
    df["Stock_vs_QQQ_3D"] = df["Log_Return_3D"] - df["QQQ_Log_Return_3D"]
    df["Stock_vs_QQQ_5D"] = df["Log_Return_5D"] - df["QQQ_Log_Return_5D"]
    return df


# REAKCJA SPÓŁKI WZGLĘDEM BENCHMARKU
def add_benchmark_event_metrics( df: pd.DataFrame) -> pd.DataFrame:

    qqq_cutoff_price = pd.to_numeric(
        df["QQQ_Cutoff_Adj_Close"],
        errors="coerce",
    )

    qqq_event_price = pd.to_numeric(
        df["QQQ_Event_Adj_Close"],
        errors="coerce",
    )

    valid_mask = (
        df["Event_Return_1D"].notna()
        & qqq_cutoff_price.notna()
        & qqq_event_price.notna()
        & np.isfinite(qqq_cutoff_price)
        & np.isfinite(qqq_event_price)
        & (qqq_cutoff_price > 0)
        & (qqq_event_price > 0)
    )

    df["QQQ_Event_Return_1D"] = np.nan

    df.loc[valid_mask, "QQQ_Event_Return_1D"] = (
        qqq_event_price[valid_mask] / qqq_cutoff_price[valid_mask] - 1
    )

    df["Abnormal_Event_Return_1D"] = df["Event_Return_1D"]- df["QQQ_Event_Return_1D"]

    return df



# TARGET
def create_target(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    cutoff_price = pd.to_numeric(
        df["Cutoff_Adj_Close"],
        errors="coerce",
    )

    event_price = pd.to_numeric(
        df["Event_Adj_Close"],
        errors="coerce",
    )

    # Walidacja ceny
    valid_price_mask = (
        cutoff_price.notna()
        & event_price.notna()
        & np.isfinite(cutoff_price)
        & np.isfinite(event_price)
        & (cutoff_price > 0)
        & (event_price > 0)
        & (df["Publication_Period"] != "INTRADAY")
    )

    # EVENT RETURN
    df["Event_Return_1D"] = np.nan

    df.loc[valid_price_mask, "Event_Return_1D"] = (
        event_price[valid_price_mask] / cutoff_price[valid_price_mask] - 1
    )

    # TARGET BINARNY
    # Domyślnie NA, 0/1 przypisujemy tylko tam, gdzie mamy policzony Event_Return_1D

    target = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Int64",
    )

    target.loc[valid_price_mask] = (
        df.loc[valid_price_mask, "Event_Return_1D"]
        .gt(0)
        .astype(int)
    )

    df["Target_Event_1D"] = target

    return df


def create_abnormal_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Abnormal_Event_Return_1D" not in df.columns:
        raise KeyError(
            "Brak kolumny: Abnormal_Event_Return_1D "
        )

    df["Target_Abnormal_1D"] = pd.NA

    abnormal_return = pd.to_numeric(
        df["Abnormal_Event_Return_1D"],
        errors="coerce",
    )

    valid_mask = (
        abnormal_return.notna()
        & np.isfinite(abnormal_return)
        & (df["Publication_Period"] != "INTRADAY")
    )

    df.loc[
        valid_mask,
        "Target_Abnormal_1D"
    ] = (
        df.loc[
            valid_mask,
            "Abnormal_Event_Return_1D"
        ] > 0
    ).astype(int)

    df["Target_Abnormal_1D"] = (
        df["Target_Abnormal_1D"]
        .astype("Int64")
    )

    return df


# Walidacja czasowo końcowego datasetu
def validate_temporal_order(df: pd.DataFrame,) -> None:

    df = df.copy()

    cutoff = pd.to_datetime(
        df["Feature_Cutoff_Session"],
        errors="coerce",
    )

    event = pd.to_datetime(
        df["Event_Session"],
        errors="coerce",
    )

    missing_mask = (
        cutoff.isna()
        | event.isna()
    )

    if missing_mask.any():
        invalid_rows = df.loc[
            missing_mask,
            ["Ticker", "Accession", "Feature_Cutoff_Session", "Event_Session"],
        ]

        raise ValueError(
            "Brak wymaganej daty sesji:\n"
            f"{invalid_rows.to_string(index=False)}"
        )

    invalid_mask = cutoff >= event

    if invalid_mask.any():
        invalid_rows = df.loc[
            invalid_mask,
            ["Ticker", "Accession", "Publication_Period", "Feature_Cutoff_Session", "Event_Session"]
        ]

        raise ValueError(
            "Feature_Cutoff_Session musi być "
            "wcześniejsza niż Event_Session:\n"
            f"{invalid_rows.to_string(index=False)}"
        )


###################################
# Budowa datasetu
###################################
def build_model_dataset(
    sec_df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> pd.DataFrame:

    result = sec_df.copy()
    market_work = market_df.copy()

    sec_date_columns = [
        "Filing_Date",
        "Feature_Cutoff_Session",
        "Event_Session",
    ]

    for column in sec_date_columns:
        result[column] = pd.to_datetime(
            result[column],
            errors="raise",
        ).dt.normalize()

    market_work["Date"] = pd.to_datetime(
        market_work["Date"],
        errors="raise",
    ).dt.normalize()

    validate_sec_data(result)
    validate_market_data(market_work)

    result = add_sentiment_momentum(result)


    # CECHY SPÓŁKI Z CUTOFF
    result = merge_cutoff_market_features(
        sec_df=result,
        market_df=market_work,
    )

    # CENA SPÓŁKI Z EVENT SESSION
    result = merge_event_price(
        df=result,
        market_df=market_work,
    )


    # QQQ Z FEATURE CUTOFF SESSION
    result = merge_benchmark_cutoff_features(
        df=result,
        market_df=market_work,
    )

    # QQQ Z EVENT SESSION
    result = merge_benchmark_event_price(
        df=result,
        market_df=market_work,
    )

    # WALIDACJA MERGE
    validate_market_matches(result)

    # RELATYWNE CECHY RYNKOWE
    result = add_relative_market_features(result)

    # TARGET SPÓŁKI
    result = create_target(result)

    # REAKCJA WZGLĘDEM QQQ
    result = add_benchmark_event_metrics(result)

    # TARGET ABNORMAL RETURN
    result = create_abnormal_target(result)

    # Nie usuwamy INTERDAY, ale zostawimay bez udzialu
    result["Use_In_Primary_Model"] = (
        result["Publication_Period"] != "INTRADAY"
    ).astype(int)


    primary_mask = result["Use_In_Primary_Model"] == 1

    if result.loc[primary_mask, "Target_Abnormal_1D"].isna().any():
        raise ValueError(
            "Znaleziono obserwację primary model bez Target_Abnormal_1D"
        )

    validate_temporal_order(result)

    for column in sec_date_columns:
        result[column] = pd.to_datetime(
            result[column],
            errors="raise"
        ).dt.date

    result.sort_values(
        by=["Ticker", "Filing_Date", "Accession"],
        inplace=True,
        ignore_index=True,
    )

    return result



def main() -> None:

    if not SEC_INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku: {SEC_INPUT_FILE}"
        )

    if not MARKET_INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku: {MARKET_INPUT_FILE}"
        )

    # wczytanie danych
    df_sec = pd.read_csv(SEC_INPUT_FILE)
    df_market = pd.read_csv(MARKET_INPUT_FILE)

    logger.info(
        "Wczytano %d filingów SEC i %d obserwacji rynkowych.",
        len(df_sec),
        len(df_market),
    )

  
    # Budowa datasetu
    

    df_model = build_model_dataset(
        sec_df=df_sec,
        market_df=df_market,
    )

   
    # Walidacja 
    if len(df_model) != len(df_sec):
        raise ValueError(
            "Budowa datasetu zmieniła liczbę filingów: "
            f"SEC={len(df_sec)}, model={len(df_model)}."
        )

    primary_mask = df_model["Use_In_Primary_Model"] == 1
    primary_count = int(primary_mask.sum())

    primary_target_counts = (
        df_model.loc[primary_mask, "Target_Abnormal_1D"]
        .value_counts(dropna=False)
        .sort_index()
    )

    logger.info(
        "Dataset: %d filingów, primary model: %d filingów.",
        len(df_model),
        primary_count,
    )

    logger.info(
        "Rozkład Target_Abnormal_1D w primary model:\n%s",
        primary_target_counts.to_string(),
    )

    # Zapisywanie
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df_model.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    logger.info(
        "Zapisano dataset modelowy do %s",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    main()