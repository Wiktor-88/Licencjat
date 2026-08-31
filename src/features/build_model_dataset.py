# Plik siódmy - tworzenie datasetu danych dla modeli ML

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

# Cechy rolling Z-score
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


# Dodanie cech benchmarku
def merge_benchmark_cutoff_features(df: pd.DataFrame, market_df: pd.DataFrame,) -> pd.DataFrame:
    """Dodaje stan benchmarku QQQ z sesji Feature_Cutoff_Session"""

    required_columns = ["Ticker", "Date", "Adj_Close"] + BENCHMARK_FEATURE_COLUMNS
    missing_columns = [col for col in required_columns if col not in market_df.columns]

    if missing_columns:
        raise ValueError(f"Brakuje wymaganych kolumn: {missing_columns}")

    market_df = market_df.copy()
    market_df["Date"] = pd.to_datetime(market_df["Date"], errors="raise")

    df = df.copy()
    df["Feature_Cutoff_Session"] = pd.to_datetime(df["Feature_Cutoff_Session"], errors="raise")

    benchmark_df = market_df.loc[market_df["Ticker"] == BENCHMARK_TICKER,
                                 ["Date", "Adj_Close"] + BENCHMARK_FEATURE_COLUMNS].copy()

    if benchmark_df.empty:
        raise ValueError(f"Nie znaleziono benchmarku {BENCHMARK_TICKER}")

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

    missing_benchmark = result["QQQ_Cutoff_Adj_Close"].isna()

    if missing_benchmark.any():
        missing_rows = result.loc[missing_benchmark,["Ticker", "Accession", "Feature_Cutoff_Session"]]

        raise ValueError(f"Brak danych QQQ dla części Feature_Cutoff_Session:\n{missing_rows.to_string(index=False)}")

    return result


# Walidacja danych SEC
def validate_sec_data(df: pd.DataFrame) -> None:

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

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Brakuje wymaganych kolumn: {missing_columns}")

    # Każdy filing powinien występować dokładnie raz
    duplicate_mask = df.duplicated(subset=["Ticker", "Accession"], keep=False,)

    if duplicate_mask.any():
        duplicates = df.loc[duplicate_mask, ["Ticker", "Accession"]]

        raise ValueError(f"Znaleziono zduplikowane filingi:\n{duplicates.to_string(index=False)}")

    # Kluczowe kolumny potrzebne do budowy datasetu
    critical_columns = [
        "Ticker",
        "Accession",
        "Publication_Period",
        "Feature_Cutoff_Session",
        "Event_Session"
    ]

    missing_critical = df[critical_columns].isna().any(axis=1)

    if missing_critical.any():
        invalid_rows = df.loc[missing_critical,
                              ["Ticker", "Accession", "Publication_Period",
                                "Feature_Cutoff_Session", "Event_Session", "Mean_Net_Sentiment"]]

        raise ValueError("Znaleziono filing z brakującymi danymi wymaganymi do budowy datasetu:\n"
                         f"{invalid_rows.to_string(index=False)}")

    valid_periods = {
        "PRE_MARKET",
        "INTRADAY",
        "AFTER_HOURS",
        "NON_TRADING_DAY",
    }

    unexpected_periods = set(df["Publication_Period"].unique()) - valid_periods

    if unexpected_periods:
        raise ValueError(f"Niepoprawne wartości: {unexpected_periods}")

    cutoff_dates = pd.to_datetime(df["Feature_Cutoff_Session"], errors="raise")

    event_dates = pd.to_datetime(df["Event_Session"], errors="raise")

    if (cutoff_dates >= event_dates).any():
        raise ValueError("Feature_Cutoff_Session musi być wcześniejsza niż Event_Session")



# sprawdzamy dane rynkowe
def validate_market_features(df: pd.DataFrame) -> None:

    required_columns = ["Ticker", "Date", "Adj_Close"] + MARKET_FEATURE_COLUMNS + ROLLING_Z_FEATURE_COLUMNS
    
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Brakuje wymaganych kolumn w market_features.csv: {missing_columns}")

    duplicate_mask = df.duplicated(subset=["Ticker", "Date"], keep=False,)

    if duplicate_mask.any():
        duplicates = df.loc[duplicate_mask, ["Ticker", "Date"]]

        raise ValueError("Znaleziono zduplikowane Ticker + Date:\n"
                        f"{duplicates.to_string(index=False)}")

    adj_close = pd.to_numeric(df["Adj_Close"], errors="coerce")

    invalid_price_mask = (adj_close.isna() | ~np.isfinite(adj_close) | (adj_close <= 0))

    if invalid_price_mask.any():
        invalid_rows = df.loc[invalid_price_mask, ["Ticker", "Date", "Adj_Close"]]

        raise ValueError(f"Znaleziono niepoprawne wartości:\n{invalid_rows.to_string(index=False)}")


def parse_acceptance_utc(series: pd.Series) -> pd.Series:
    """Parsuje Acceptance_DateTime_ET do UTC"""

    timestamp_text = series.astype("string").str.strip()

    has_timezone = timestamp_text.str.contains(
        r"(?:Z|[+-]\d{2}:\d{2})$",
        regex=True,
        na=False,
    )

    if not has_timezone.all():
        invalid_values = timestamp_text[~has_timezone]

        raise ValueError("Znaleziono brakujące Acceptance_DateTime_ET lub wartości bez jawnego offsetu strefy czasowej:\n"
            f"{invalid_values.to_string()}")

    return pd.to_datetime(
        timestamp_text,
        format="mixed",
        utc=True,
        errors="raise",
    )


def validate_market_matches(df: pd.DataFrame) -> None:
    required_price_columns = [
        "Cutoff_Adj_Close",
        "Event_Adj_Close",
        "QQQ_Cutoff_Adj_Close",
        "QQQ_Event_Adj_Close",
    ]

    required_columns = [
        "Ticker",
        "Accession",
        "Feature_Cutoff_Session",
        "Event_Session",
    ] + required_price_columns

    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Brakuje kolumn wymaganych do walidacji danych rynkowych: {missing_columns}")

    price_data = df[required_price_columns].apply(pd.to_numeric, errors="coerce")

    invalid_price_mask = (price_data.isna() | ~np.isfinite(price_data) | (price_data <= 0)).any(axis=1)

    if invalid_price_mask.any():
        invalid_rows = df.loc[invalid_price_mask,
            ["Ticker", "Accession", "Feature_Cutoff_Session", "Event_Session"] + required_price_columns]

        raise ValueError("Znaleziono brakujące lub niepoprawne dane cenowe spółki lub QQQ:\n"
            f"{invalid_rows.to_string(index=False)}")


# Momentum sentymentu
def add_sentiment_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Porównuje sentyment bieżącego filingu ze średnim sentymentem maksymalnie 3 wcześniejszych filingów
    tej samej spółki
    """

    df = df.copy()
    df["_Acceptance_UTC"] = parse_acceptance_utc(df["Acceptance_DateTime_ET"])

    # Ten sam czas publikacji uniemożliwia ustalenie kolejności
    same_time_mask = df.duplicated(subset=["Ticker", "_Acceptance_UTC"], keep=False)

    if same_time_mask.any():
        same_time_rows = df.loc[same_time_mask, ["Ticker", "Accession", "_Acceptance_UTC"]]

        raise ValueError("Znaleziono kilka filingów tego samego tickera z identycznym Acceptance_DateTime_ET:\n"
            f"{same_time_rows.to_string(index=False)}")

    df = df.sort_values(["Ticker", "_Acceptance_UTC", "Accession"]).reset_index(drop=True)

    df["Previous_Sentiment_Mean_3"] = np.nan
    df["Sentiment_History_Count_3"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

    # Do historii trafiają tylko filingi z dostępnym sentymentem
    valid_mask = df["Mean_Net_Sentiment"].notna()

    sentiment_history = df.loc[valid_mask, ["Ticker", "Mean_Net_Sentiment"]].copy()

    grouped = sentiment_history.groupby("Ticker", sort=False)["Mean_Net_Sentiment"]

    sentiment_history["Previous_Sentiment_Mean_3"] = grouped.transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean())

    sentiment_history["Sentiment_History_Count_3"] = grouped.transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).count()).astype("Int64")

    # Liczenie momentum
    df.loc[sentiment_history.index, "Previous_Sentiment_Mean_3"] = sentiment_history["Previous_Sentiment_Mean_3"]

    df.loc[sentiment_history.index, "Sentiment_History_Count_3"] = sentiment_history["Sentiment_History_Count_3"]

    df["Sentiment_Momentum_3"] = df["Mean_Net_Sentiment"] - df["Previous_Sentiment_Mean_3"]
    
    return df.drop(columns="_Acceptance_UTC")


# Łączenie cech rynkowych z feature cutoff season
def merge_cutoff_market_features(sec_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:

    cutoff_market = market_df[["Ticker", "Date", "Adj_Close"] + MARKET_FEATURE_COLUMNS + ROLLING_Z_FEATURE_COLUMNS].copy()

    cutoff_market = cutoff_market.rename(columns={
        "Date": "Feature_Cutoff_Session",
        "Adj_Close": "Cutoff_Adj_Close",
    })

    return sec_df.merge(
        cutoff_market,
        on=["Ticker", "Feature_Cutoff_Session"],
        how="left",
        validate="many_to_one",
    )


# Dodanie cechy z event session
def merge_event_price(df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:

    event_prices = market_df[["Ticker", "Date", "Adj_Close"]].copy()

    event_prices = event_prices.rename(columns={
        "Date": "Event_Session",
        "Adj_Close": "Event_Adj_Close",
    })

    return df.merge(
        event_prices,
        on=["Ticker", "Event_Session"],
        how="left",
        validate="many_to_one",
    )


# DODANIE CENY BENCHMARKU Z EVENT SESSION
def merge_benchmark_event_price(df: pd.DataFrame, market_df: pd.DataFrame,) -> pd.DataFrame:

    benchmark_prices = market_df.loc[market_df["Ticker"] == BENCHMARK_TICKER, ["Date", "Adj_Close"]].copy()

    if benchmark_prices.empty:
        raise ValueError(f"Nie znaleziono benchmarku {BENCHMARK_TICKER}")

    benchmark_prices = benchmark_prices.rename(columns={
        "Date": "Event_Session",
        "Adj_Close": "QQQ_Event_Adj_Close",
    })

    return df.merge(
        benchmark_prices,
        on="Event_Session",
        how="left",
        validate="many_to_one",
    )


# Relatywne cechy vs QQQ
def add_relative_market_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Stock_vs_QQQ_1D"] = df["Log_Return_1D"] - df["QQQ_Log_Return_1D"]
    df["Stock_vs_QQQ_3D"] = df["Log_Return_3D"] - df["QQQ_Log_Return_3D"]
    df["Stock_vs_QQQ_5D"] = df["Log_Return_5D"] - df["QQQ_Log_Return_5D"]

    return df


# Reakcja spółki względem Benchmarku
def add_benchmark_event_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["QQQ_Event_Return_1D"] = df["QQQ_Event_Adj_Close"] / df["QQQ_Cutoff_Adj_Close"] - 1
    df["Abnormal_Event_Return_1D"] =  df["Event_Return_1D"] - df["QQQ_Event_Return_1D"]

    return df



# Tworzenie Targetu 
def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    valid_mask = df["Publication_Period"] != "INTRADAY"

    df["Event_Return_1D"] = np.nan
    df.loc[valid_mask, "Event_Return_1D"] = (
        df.loc[valid_mask, "Event_Adj_Close"] / df.loc[valid_mask, "Cutoff_Adj_Close"] - 1)

    target = pd.Series(pd.NA, index=df.index, dtype="Int64")
    target.loc[valid_mask] = (df.loc[valid_mask, "Event_Return_1D"] > 0).astype(int)

    df["Target_Event_1D"] = target

    return df


def create_abnormal_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Abnormal_Event_Return_1D" not in df.columns:
        raise KeyError("Brak kolumny: Abnormal_Event_Return_1D")

    abnormal_return = pd.to_numeric(
        df["Abnormal_Event_Return_1D"],
        errors="coerce",
    )

    valid_mask = (
        abnormal_return.notna()
        & np.isfinite(abnormal_return)
        & (df["Publication_Period"] != "INTRADAY")
    )

    target = pd.Series(pd.NA, index=df.index, dtype="Int64")
    target.loc[valid_mask] = (
        abnormal_return.loc[valid_mask] > 0
    ).astype(int)

    df["Target_Abnormal_1D"] = target

    return df


# Walidacja czasowo końcowego datasetu
def validate_temporal_order(df: pd.DataFrame,) -> None:

    df = df.copy()

    cutoff = pd.to_datetime(df["Feature_Cutoff_Session"], errors="coerce")

    event = pd.to_datetime(df["Event_Session"], errors="coerce")

    missing_mask = (cutoff.isna() | event.isna())

    if missing_mask.any():
        invalid_rows = df.loc[missing_mask, ["Ticker", "Accession", "Feature_Cutoff_Session", "Event_Session"]]

        raise ValueError(f"Brak wymaganej daty sesji:\n{invalid_rows.to_string(index=False)}")

    invalid_mask = cutoff >= event

    if invalid_mask.any():
        invalid_rows = df.loc[invalid_mask,
            ["Ticker", "Accession", "Publication_Period", "Feature_Cutoff_Session", "Event_Session"]]

        raise ValueError("Feature_Cutoff_Session musi być wcześniejsza niż Event_Session:\n"
            f"{invalid_rows.to_string(index=False)}")


# Tworzenie końcowej ramki danych
def build_model_dataset(sec_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:

    result = sec_df.copy()
    market_work = market_df.copy()

    # Walidacja danych
    validate_sec_data(result)
    validate_market_features(market_work)

    # Formatowanie dat
    sec_date_columns = [
        "Filing_Date",
        "Feature_Cutoff_Session",
        "Event_Session",
    ]

    for column in sec_date_columns:
        result[column] = pd.to_datetime(result[column], errors="raise").dt.normalize()

    market_work["Date"] = pd.to_datetime(market_work["Date"], errors="raise").dt.normalize()

    # Dodadawanie cech
    # Momentum
    result = add_sentiment_momentum(result)

    # Cechy spółki z Cutoff
    result = merge_cutoff_market_features(result, market_work)

    # Cena spółki z Session
    result = merge_event_price(result, market_work)

    # QQQ z feature cutoff season
    result = merge_benchmark_cutoff_features(result, market_work)

    # QQQ z event season
    result = merge_benchmark_event_price(result, market_work)

    # walidacja złączenia ramek danych
    validate_market_matches(result)

    # relatywne cechy
    result = add_relative_market_features(result)

    # target
    result = create_target(result)

    # rekacja wzgledem qqq
    result = add_benchmark_event_metrics(result)

    # target abnormal
    result = create_abnormal_target(result)

    # Pominięcie INTERDAY
    has_sentiment = result["Mean_Net_Sentiment"].notna()

    result["Use_In_Primary_Model"] = ((result["Publication_Period"] != "INTRADAY")
                                    & has_sentiment).astype(int)
    primary_mask = result["Use_In_Primary_Model"] == 1

    # Sprawdzenie czy kazda obserwacja ma y
    if result.loc[primary_mask, "Target_Abnormal_1D"].isna().any():
        raise ValueError("Znaleziono obserwację primary model bez Target_Abnormal_1D")

    validate_temporal_order(result)

    # Format dat do zapisu CSV
    for column in sec_date_columns:
        result[column] = pd.to_datetime(result[column], errors="raise").dt.date

    return result.sort_values(["Ticker", "Filing_Date", "Accession"]).reset_index(drop=True)
    



def main() -> None:

    if not SEC_INPUT_FILE.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {SEC_INPUT_FILE}")

    if not MARKET_INPUT_FILE.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {MARKET_INPUT_FILE}")

    # Wczytanie danych
    df_sec = pd.read_csv(SEC_INPUT_FILE)
    df_market = pd.read_csv(MARKET_INPUT_FILE)

    logger.info("Wczytano %d filingów SEC i %d obserwacji rynkowych", len(df_sec), len(df_market))

    # Tworzenie ramki danych
    df_model = build_model_dataset(sec_df=df_sec, market_df=df_market,)


    missing_sentiment_count = int(df_model["Mean_Net_Sentiment"].isna().sum())
    logger.info("Filingi bez dostępnego sentymentu : %d", missing_sentiment_count)

    # Sprawdzenie
    if len(df_model) != len(df_sec):
        raise ValueError(f"Budowa datasetu zmieniła liczbę filingów: SEC={len(df_sec)}, model={len(df_model)}")

    primary_mask = df_model["Use_In_Primary_Model"] == 1
    primary_count = int(primary_mask.sum())
    excluded_count = len(df_model) - primary_count

    primary_target_counts = df_model.loc[primary_mask, "Target_Abnormal_1D"].value_counts(dropna=False).sort_index()

    logger.info("Dataset: %d filingów, primary model: %d, wyłączone: %d.", len(df_model), primary_count, excluded_count)

    logger.info("Rozkład Target_Abnormal_1D w primary model:\n%s", primary_target_counts.to_string())

    # Zapis
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_model.to_csv(OUTPUT_FILE, index=False)

    logger.info("Zapisano dataset modelowy do %s", OUTPUT_FILE)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    main()