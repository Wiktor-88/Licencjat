from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# ŚCIEŻKI
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEC_INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sec_event_features.csv"
)

MARKET_INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "market_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_dataset.csv"
)


# ============================================================
# CECHY RYNKOWE UŻYWANE W MODELU
# ============================================================

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

############### ROLLING Z SCORE FEATURES ###############

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



# ============================================================
# BENCHMARK RYNKOWY
# ============================================================

BENCHMARK_TICKER = "QQQ"

BENCHMARK_FEATURE_COLUMNS = [
    "Log_Return_1D",
    "Log_Return_3D",
    "Log_Return_5D",
    "Volatility_14D",
]


# ============================================================
# DODANIE CECH BENCHMARKU Z FEATURE CUTOFF SESSION
# ============================================================

def merge_benchmark_cutoff_features(
    df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> pd.DataFrame:

    benchmark_df = market_df.loc[
        market_df["Ticker"] == BENCHMARK_TICKER,
        [
            "Date",
            "Adj_Close",
        ] + BENCHMARK_FEATURE_COLUMNS,
    ].copy()

    if benchmark_df.empty:
        raise ValueError(
            f"Nie znaleziono benchmarku {BENCHMARK_TICKER} "
            "w market_features.csv."
        )

    benchmark_df = benchmark_df.rename(
        columns={
            "Date": "Feature_Cutoff_Session",
            "Adj_Close": "QQQ_Cutoff_Adj_Close",
            "Log_Return_1D": "QQQ_Log_Return_1D",
            "Log_Return_3D": "QQQ_Log_Return_3D",
            "Log_Return_5D": "QQQ_Log_Return_5D",
            "Volatility_14D": "QQQ_Volatility_14D",
        }
    )

    result = df.merge(
        benchmark_df,
        on="Feature_Cutoff_Session",
        how="left",
        validate="many_to_one",
    )

    return result


# ============================================================
# WALIDACJA DANYCH SEC
# ============================================================

def validate_sec_data(
    df: pd.DataFrame,
) -> None:

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
            "Brakuje wymaganych kolumn "
            "w sec_event_features.csv:\n"
            + "\n".join(missing_columns)
        )

    # Każdy accession powinien występować
    # dokładnie raz.
    duplicate_mask = df.duplicated(
        subset=[
            "Ticker",
            "Accession",
        ],
        keep=False,
    )

    if duplicate_mask.any():

        duplicates = df.loc[
            duplicate_mask,
            [
                "Ticker",
                "Accession",
            ],
        ]

        raise ValueError(
            "Znaleziono zduplikowane filingi:\n"
            f"{duplicates}"
        )


# ============================================================
# WALIDACJA DANYCH RYNKOWYCH
# ============================================================

def validate_market_data(
    df: pd.DataFrame,
) -> None:

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
            + "\n".join(missing_columns)
        )

    duplicate_mask = df.duplicated(
        subset=[
            "Ticker",
            "Date",
        ],
        keep=False,
    )

    if duplicate_mask.any():

        duplicates = df.loc[
            duplicate_mask,
            [
                "Ticker",
                "Date",
            ],
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

    invalid_price_mask = (
        adj_close.isna()
        | ~np.isfinite(adj_close)
        | (adj_close <= 0)
    )

    if invalid_price_mask.any():
        invalid_rows = df.loc[
            invalid_price_mask,
            [
                "Ticker",
                "Date",
                "Adj_Close",
            ],
        ]

        raise ValueError(
            "Znaleziono niepoprawne wartości "
            "Adj_Close:\n"
            f"{invalid_rows}"
        )


def parse_acceptance_utc(series: pd.Series) -> pd.Series:
    """
    Parsuje Acceptance_DateTime_ET do UTC.

    Wymagamy jawnej informacji o strefie czasowej,
    np.:
        2026-05-20T16:21:19-04:00

    Nie pozwalamy na naive datetime, ponieważ
    nie chcemy zgadywać strefy czasowej.
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
            "Znaleziono Acceptance_DateTime_ET bez "
            "jawnego offsetu strefy czasowej:\n"
            f"{invalid_values.to_string()}"
        )

    return pd.to_datetime(
        timestamp_text,
        format="mixed",
        utc=True,
        errors="raise",
    )



########## WALIDACJA MARKET MATCHES ###############
def validate_market_matches(
    df: pd.DataFrame,
) -> None:

    required_price_columns = [
        "Cutoff_Adj_Close",
        "Event_Adj_Close",
        "QQQ_Cutoff_Adj_Close",
        "QQQ_Event_Adj_Close",
    ]

    missing_mask = (
        df[required_price_columns]
        .isna()
        .any(axis=1)
    )

    if missing_mask.any():
        missing_rows = df.loc[
            missing_mask,
            [
                "Ticker",
                "Accession",
                "Feature_Cutoff_Session",
                "Event_Session",
            ] + required_price_columns,
        ]

        raise ValueError(
            "Nie znaleziono wszystkich wymaganych "
            "danych rynkowych spółki lub QQQ:\n"
            f"{missing_rows}"
        )






# ============================================================
# MOMENTUM SENTYMENTU
# ============================================================

def add_sentiment_momentum(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Liczy różnicę między sentymentem bieżącego
    filingu a średnim sentymentem maksymalnie
    3 wcześniejszych filingów z dostępnym sentymentem.

    Finingi bez Mean_Net_Sentiment nie są traktowane
    jako neutralne i nie wchodzą do historii sentymentu.
    """

    df["_Acceptance_UTC"] = parse_acceptance_utc(
        df["Acceptance_DateTime_ET"]
    )

    df.sort_values(
        by=[
            "Ticker",
            "_Acceptance_UTC",
            "Accession",
        ],
        inplace=True,
        ignore_index=True,
    )

    df["Previous_Sentiment_Mean_3"] = np.nan

    df["Sentiment_History_Count_3"] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Int64",
    )

    # ========================================================
    # TYLKO FILINGI Z DOSTĘPNYM SENTYMENTEM
    # ========================================================

    valid_mask = df[
        "Mean_Net_Sentiment"
    ].notna()

    sentiment_history = df.loc[
        valid_mask,
        [
            "Ticker",
            "Mean_Net_Sentiment",
        ],
    ].copy()

    # ========================================================
    # POPRZEDNI SENTYMENT
    # ========================================================
    #
    # shift(1) gwarantuje, że bieżący filing
    # nie trafia do własnej historii.
    # ========================================================

    shifted_sentiment = (
        sentiment_history
        .groupby(
            "Ticker",
            sort=False,
        )["Mean_Net_Sentiment"]
        .shift(1)
    )

    # ========================================================
    # ŚREDNIA Z MAKSYMALNIE 3 WCZEŚNIEJSZYCH
    # ========================================================

    previous_mean = (
        shifted_sentiment
        .groupby(
            sentiment_history["Ticker"],
            sort=False,
        )
        .rolling(
            window=3,
            min_periods=1,
        )
        .mean()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    previous_count = (
        shifted_sentiment
        .groupby(
            sentiment_history["Ticker"],
            sort=False,
        )
        .rolling(
            window=3,
            min_periods=1,
        )
        .count()
        .reset_index(
            level=0,
            drop=True,
        )
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

    # ========================================================
    # MOMENTUM SENTYMENTU
    # ========================================================

    df["Sentiment_Momentum_3"] = (
        df["Mean_Net_Sentiment"]
        - df["Previous_Sentiment_Mean_3"]
    )

    df.drop(
        columns=["_Acceptance_UTC"],
        inplace=True,
    )

    return df


# ============================================================
# ŁĄCZENIE CECH RYNKOWYCH Z FEATURE CUTOFF SESSION
# ============================================================

def merge_cutoff_market_features(
    sec_df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> pd.DataFrame:

    cutoff_market = market_df[
        [
            "Ticker",
            "Date",
            "Adj_Close",
        ]
        + MARKET_FEATURE_COLUMNS + ROLLING_Z_FEATURE_COLUMNS
    ].copy()

    cutoff_market = cutoff_market.rename(
        columns={
            "Date":
                "Feature_Cutoff_Session",

            "Adj_Close":
                "Cutoff_Adj_Close",
        }
    )

    result = sec_df.merge(
        cutoff_market,
        on=[
            "Ticker",
            "Feature_Cutoff_Session",
        ],
        how="left",
        validate="many_to_one",
    )

    return result


# ============================================================
# DODANIE CENY Z EVENT SESSION
# ============================================================

def merge_event_price(
    df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> pd.DataFrame:

    event_prices = market_df[
        [
            "Ticker",
            "Date",
            "Adj_Close",
        ]
    ].copy()

    event_prices = event_prices.rename(
        columns={
            "Date":
                "Event_Session",

            "Adj_Close":
                "Event_Adj_Close",
        }
    )

    result = df.merge(
        event_prices,
        on=[
            "Ticker",
            "Event_Session",
        ],
        how="left",
        validate="many_to_one",
    )

    return result


# ============================================================
# DODANIE CENY BENCHMARKU Z EVENT SESSION
# ============================================================

def merge_benchmark_event_price(
    df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> pd.DataFrame:

    benchmark_prices = market_df.loc[
        market_df["Ticker"] == BENCHMARK_TICKER,
        [
            "Date",
            "Adj_Close",
        ],
    ].copy()

    if benchmark_prices.empty:
        raise ValueError(
            f"Nie znaleziono benchmarku {BENCHMARK_TICKER} "
            "w market_features.csv."
        )

    benchmark_prices = benchmark_prices.rename(
        columns={
            "Date": "Event_Session",
            "Adj_Close": "QQQ_Event_Adj_Close",
        }
    )

    result = df.merge(
        benchmark_prices,
        on="Event_Session",
        how="left",
        validate="many_to_one",
    )

    return result


# ============================================================
# RELATYWNE CECHY SPÓŁKA VS QQQ
# ============================================================

def add_relative_market_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df["Stock_vs_QQQ_1D"] = (
        df["Log_Return_1D"]
        - df["QQQ_Log_Return_1D"]
    )

    df["Stock_vs_QQQ_3D"] = (
        df["Log_Return_3D"]
        - df["QQQ_Log_Return_3D"]
    )

    df["Stock_vs_QQQ_5D"] = (
        df["Log_Return_5D"]
        - df["QQQ_Log_Return_5D"]
    )

    return df


# ============================================================
# REAKCJA SPÓŁKI WZGLĘDEM BENCHMARKU
# ============================================================

def add_benchmark_event_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:

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

    df.loc[
        valid_mask,
        "QQQ_Event_Return_1D",
    ] = (
        qqq_event_price[valid_mask]
        / qqq_cutoff_price[valid_mask]
        - 1
    )

    df["Abnormal_Event_Return_1D"] = (
        df["Event_Return_1D"]
        - df["QQQ_Event_Return_1D"]
    )

    return df





# ============================================================
# TARGET
# ============================================================

def create_target(
    df: pd.DataFrame,
) -> pd.DataFrame:

    cutoff_price = pd.to_numeric(
        df["Cutoff_Adj_Close"],
        errors="coerce",
    )

    event_price = pd.to_numeric(
        df["Event_Adj_Close"],
        errors="coerce",
    )

    # ========================================================
    # POPRAWNE CENY
    # ========================================================

    valid_price_mask = (
        cutoff_price.notna()
        & event_price.notna()
        & np.isfinite(cutoff_price)
        & np.isfinite(event_price)
        & (cutoff_price > 0)
        & (event_price > 0)
        & (df["Publication_Period"] != "INTRADAY")
    )

    # ========================================================
    # EVENT RETURN
    # ========================================================

    df["Event_Return_1D"] = np.nan

    df.loc[
        valid_price_mask,
        "Event_Return_1D",
    ] = (
        event_price[valid_price_mask]
        / cutoff_price[valid_price_mask]
        - 1
    )

    # ========================================================
    # TARGET BINARNY
    # ========================================================
    #
    # Domyślnie NA.
    # 0/1 przypisujemy tylko tam, gdzie naprawdę
    # mamy poprawnie policzony Event_Return_1D.
    # ========================================================

    target = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Int64",
    )

    target.loc[valid_price_mask] = (
        df.loc[
            valid_price_mask,
            "Event_Return_1D",
        ]
        .gt(0)
        .astype(int)
    )


    df["Target_Event_1D"] = target

    return df




def create_abnormal_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Abnormal_Event_Return_1D" not in df.columns:
        raise KeyError(
            "Brak kolumny 'Abnormal_Event_Return_1D'. "
            "Target abnormal musi być tworzony dopiero po obliczeniu "
            "zwrotu akcji i benchmarku QQQ."
        )

    df["Target_Abnormal_1D"] = pd.NA

    valid_mask = (
        df["Abnormal_Event_Return_1D"].notna()
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





# ============================================================
# WALIDACJA CZASOWA FINALNEGO DATASETU
# ============================================================

def validate_temporal_order(
    df: pd.DataFrame,
) -> None:

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
            [
                "Ticker",
                "Accession",
                "Feature_Cutoff_Session",
                "Event_Session",
            ],
        ]

        raise ValueError(
            "Brak wymaganej daty sesji:\n"
            f"{invalid_rows}"
        )

    invalid_mask = cutoff >= event

    if invalid_mask.any():
        invalid_rows = df.loc[
            invalid_mask,
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
            "wcześniejsza niż Event_Session:\n"
            f"{invalid_rows}"
        )


# ============================================================
# BUDOWA DATASETU
# ============================================================

def build_model_dataset(
    sec_df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> pd.DataFrame:

    # Jedna świadoma kopia robocza.
    result = sec_df.copy()
    market_work = market_df.copy()

    sec_date_columns = [
        "Filing_Date",
        "Feature_Cutoff_Session",
        "Event_Session",
    ]

    for column in sec_date_columns:
        result[column] = pd.to_datetime(
            result[column]
        )

    market_work["Date"] = pd.to_datetime(
        market_work["Date"]
    )

    validate_sec_data(result)
    validate_market_data(market_work)

    result = add_sentiment_momentum(result)

    # ========================================================
    # CECHY SPÓŁKI Z CUTOFF
    # ========================================================

    result = merge_cutoff_market_features(
        sec_df=result,
        market_df=market_work,
    )

    # ========================================================
    # CENA SPÓŁKI Z EVENT SESSION
    # ========================================================

    result = merge_event_price(
        df=result,
        market_df=market_work,
    )

    # ========================================================
    # QQQ Z FEATURE CUTOFF SESSION
    # ========================================================

    result = merge_benchmark_cutoff_features(
        df=result,
        market_df=market_work,
    )

    # ========================================================
    # QQQ Z EVENT SESSION
    # ========================================================

    result = merge_benchmark_event_price(
        df=result,
        market_df=market_work,
    )

    # ========================================================
    # WALIDACJA MERGE
    # ========================================================

    validate_market_matches(result)

    # ========================================================
    # RELATYWNE CECHY RYNKOWE
    # ========================================================

    result = add_relative_market_features(result)

    # ========================================================
    # TARGET SPÓŁKI
    # ========================================================

    result = create_target(result)

    # ========================================================
    # REAKCJA WZGLĘDEM QQQ
    # ========================================================

    result = add_benchmark_event_metrics(result)

    # ========================================================
    # TARGET ABNORMAL RETURN
    # ========================================================

    result = create_abnormal_target(result)


    result["Use_In_Primary_Model"] = (
        result["Publication_Period"] != "INTRADAY"
    ).astype(int)

    validate_temporal_order(result)

    for column in sec_date_columns:
        result[column] = pd.to_datetime(
            result[column]
        ).dt.date

    result.sort_values(
        by=[
            "Ticker",
            "Filing_Date",
            "Accession",
        ],
        inplace=True,
        ignore_index=True,
    )

    return result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 80
    )

    print(
        "BUDOWA DATASETU MODELOWEGO"
    )

    print(
        "=" * 80
    )

    print(
        f"\nSEC:\n{SEC_INPUT_FILE}"
    )

    print(
        f"\nMarket:\n{MARKET_INPUT_FILE}"
    )

    # ========================================================
    # KONTROLA PLIKÓW
    # ========================================================

    if not SEC_INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku:\n"
            f"{SEC_INPUT_FILE}"
        )

    if not MARKET_INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku:\n"
            f"{MARKET_INPUT_FILE}"
        )

    # ========================================================
    # WCZYTANIE
    # ========================================================

    df_sec = pd.read_csv(
        SEC_INPUT_FILE
    )

    df_market = pd.read_csv(
        MARKET_INPUT_FILE
    )

    print(
        f"\nLiczba filingów SEC: "
        f"{len(df_sec)}"
    )

    print(
        f"Liczba obserwacji rynkowych: "
        f"{len(df_market)}"
    )

    # ========================================================
    # BUDOWA
    # ========================================================

    df_model = build_model_dataset(
        sec_df=df_sec,
        market_df=df_market,
    )

    # ========================================================
    # ZAPIS
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df_model.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # PODGLĄD
    # ========================================================

    columns_to_show = [
        "Ticker",
        "Filing_Date",
        "Publication_Period",
        "Feature_Cutoff_Session",
        "Event_Session",
        "Mean_Net_Sentiment",
        "Sentiment_Momentum_3",

        "Log_Return_1D",
        "Log_Return_3D",
        "Log_Return_5D",
        "Volatility_14D",
        "Relative_Volume_20D",
        "RSI_14",
        "Price_to_SMA20",

        "QQQ_Log_Return_1D",
        "QQQ_Log_Return_3D",
        "QQQ_Log_Return_5D",
        "QQQ_Volatility_14D",

        "Stock_vs_QQQ_1D",
        "Stock_vs_QQQ_3D",
        "Stock_vs_QQQ_5D",

        "Cutoff_Adj_Close",
        "Event_Adj_Close",
        "Event_Return_1D",

        "QQQ_Cutoff_Adj_Close",
        "QQQ_Event_Adj_Close",
        "QQQ_Event_Return_1D",

        "Abnormal_Event_Return_1D",
        "Target_Event_1D",
        "Target_Abnormal_1D",
        "Use_In_Primary_Model",
        "Accession",
    ]

    print(
        "\n"
        + "=" * 80
    )

    print(
        "WYNIK"
    )

    print(
        "=" * 80
    )

    print(
        df_model[
            columns_to_show
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # PODSUMOWANIE TARGETU
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "ROZKŁAD TARGETU"
    )

    print(
        "=" * 80
    )

    print(
        df_model[
            "Target_Event_1D"
        ].value_counts(
            dropna=False
        )
    )

    print("\nTarget_Abnormal_1D:")
    print(
        df_model["Target_Abnormal_1D"]
        .value_counts(dropna=False)
        .sort_index()
    )


    print("\nPrimary model / abnormal target audit:")

    print(
        pd.crosstab(
            df_model["Use_In_Primary_Model"],
            df_model["Target_Abnormal_1D"],
            dropna=False
        )
    )

    # ========================================================
    # PRIMARY MODEL
    # ========================================================

    primary_df = df_model[
        df_model[
            "Use_In_Primary_Model"
        ] == 1
    ]

    print(
        "\nLiczba wszystkich filingów: "
        f"{len(df_model)}"
    )

    print(
        "Liczba filingów w głównym eksperymencie: "
        f"{len(primary_df)}"
    )

    # ========================================================
    # BRAKI CECH RYNKOWYCH
    # ========================================================

    print(
        "\nBraki w cechach rynkowych:"
    )

    print(
        df_model[
            MARKET_FEATURE_COLUMNS
        ]
        .isna()
        .sum()
    )


    benchmark_feature_columns = [
        "QQQ_Log_Return_1D",
        "QQQ_Log_Return_3D",
        "QQQ_Log_Return_5D",
        "QQQ_Volatility_14D",
        "Stock_vs_QQQ_1D",
        "Stock_vs_QQQ_3D",
        "Stock_vs_QQQ_5D",
    ]

    print(
        "\nBraki w cechach benchmarku QQQ:"
    )

    print(
        df_model[
            benchmark_feature_columns
        ]
        .isna()
        .sum()
    )

    sentiment_columns = [
        "Mean_Net_Sentiment",
        "Sentiment_Momentum_3",
    ]

    print(
        "\nBraki w cechach sentymentu:"
    )

    print(
        df_model[
            sentiment_columns
        ]
        .isna()
        .sum()
    )


    print(
        "\nBraki w rolling Z-score dla filingów:"
    )

    print(
        df_model[
            ROLLING_Z_FEATURE_COLUMNS
        ]
        .isna()
        .sum()
    )


    outcome_columns = [
        "Event_Return_1D",
        "QQQ_Event_Return_1D",
        "Abnormal_Event_Return_1D",
        "Target_Event_1D",
        "Target_Abnormal_1D",
    ]

    print(
        "\nBraki w targetach / wynikach zdarzenia:"
    )

    print(
        df_model[
            outcome_columns
        ]
        .isna()
        .sum()
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