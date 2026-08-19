from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# ŚCIEŻKI
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "market_data.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "market_features.csv"
)


# ============================================================
# KOLUMNY WYMAGANE
# ============================================================

REQUIRED_COLUMNS = [
    "Ticker",
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj_Close",
    "Volume",
]


# ============================================================
# CECHY, KTÓRE PÓŹNIEJ MOŻEMY PODAĆ DO MODELU
# ============================================================

MODEL_FEATURE_COLUMNS = [
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


# ============================================================
# WALIDACJA
# ============================================================

def validate_market_data(
    df: pd.DataFrame,
) -> None:

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Brakuje wymaganych kolumn "
            "w market_data.csv:\n"
            + "\n".join(missing_columns)
        )

    # --------------------------------------------------------
    # DUPLIKATY TICKER + DATE
    # --------------------------------------------------------

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
            "Znaleziono duplikaty "
            "Ticker + Date:\n"
            f"{duplicates}"
        )

    # --------------------------------------------------------
    # BRAKI W DANYCH ŹRÓDŁOWYCH
    # --------------------------------------------------------

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj_Close",
        "Volume",
    ]

    if df[numeric_columns].isna().any().any():

        missing_counts = (
            df[numeric_columns]
            .isna()
            .sum()
        )

        raise ValueError(
            "Znaleziono braki w danych OHLCV:\n"
            f"{missing_counts}"
        )

    # --------------------------------------------------------
    # CENY MUSZĄ BYĆ DODATNIE
    # --------------------------------------------------------

    price_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj_Close",
    ]

    invalid_prices = (
        df[price_columns] <= 0
    ).any(
        axis=1
    )

    if invalid_prices.any():

        invalid_rows = df.loc[
            invalid_prices,
            [
                "Ticker",
                "Date",
            ]
            + price_columns
        ]

        raise ValueError(
            "Znaleziono niepoprawne ceny "
            "(<= 0):\n"
            f"{invalid_rows}"
        )

    # --------------------------------------------------------
    # WOLUMEN NIE MOŻE BYĆ UJEMNY
    # --------------------------------------------------------

    if (df["Volume"] < 0).any():
        raise ValueError(
            "Znaleziono ujemny wolumen."
        )


# ============================================================
# RSI
# ============================================================

def calculate_rsi(
    prices: pd.Series,
    window: int = 14,
) -> pd.Series:
    """
    RSI liczone metodą wygładzania Wildera.

    Wynik:
    0   -> bardzo słabe momentum
    100 -> bardzo silne momentum
    """

    delta = prices.diff()

    gains = delta.clip(
        lower=0
    )

    losses = (
        -delta.clip(
            upper=0
        )
    )

    average_gain = gains.ewm(
        alpha=1 / window,
        adjust=False,
        min_periods=window,
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / window,
        adjust=False,
        min_periods=window,
    ).mean()

    relative_strength = (
        average_gain
        / average_loss
    )

    rsi = (
        100
        - (
            100
            / (
                1
                + relative_strength
            )
        )
    )

    # Jeżeli przez całe okno nie było
    # ani wzrostów, ani spadków,
    # przyjmujemy neutralne RSI = 50.
    neutral_mask = (
        (average_gain == 0)
        &
        (average_loss == 0)
    )

    rsi = rsi.mask(
        neutral_mask,
        50.0,
    )

    return rsi


# ============================================================
# CECHY DLA JEDNEGO TICKERA
# ============================================================

def calculate_ticker_features(
    ticker_df: pd.DataFrame,
) -> pd.DataFrame:

    df = ticker_df.copy()

    df = df.sort_values(
        by="Date"
    ).reset_index(
        drop=True
    )

    # ========================================================
    # 1. LOGARYTMICZNE STOPY ZWROTU
    # ========================================================

    df["Log_Return_1D"] = np.log(
        df["Adj_Close"]
        /
        df["Adj_Close"].shift(1)
    )

    df["Log_Return_3D"] = np.log(
        df["Adj_Close"]
        /
        df["Adj_Close"].shift(3)
    )

    df["Log_Return_5D"] = np.log(
        df["Adj_Close"]
        /
        df["Adj_Close"].shift(5)
    )

    # ========================================================
    # 2. ZMIENNOŚĆ 14-DNIOWA
    # ========================================================

    df["Volatility_14D"] = (
        df["Log_Return_1D"]
        .rolling(
            window=14,
            min_periods=14,
        )
        .std()
    )

    # ========================================================
    # 3. ŚREDNI WOLUMEN Z POPRZEDNICH 20 SESJI
    # ========================================================
    #
    # shift(1) jest tutaj bardzo ważny.
    #
    # Średnia bazowa nie zawiera wolumenu
    # aktualnej sesji.
    # ========================================================

    df["Avg_Volume_20D_Previous"] = (
        df["Volume"]
        .shift(1)
        .rolling(
            window=20,
            min_periods=20,
        )
        .mean()
    )

    df["Relative_Volume_20D"] = (
        df["Volume"]
        /
        df["Avg_Volume_20D_Previous"]
    )

    # ========================================================
    # 4. SMA 20
    # ========================================================

    df["SMA_20"] = (
        df["Adj_Close"]
        .rolling(
            window=20,
            min_periods=20,
        )
        .mean()
    )

    df["Price_to_SMA20"] = (
        df["Adj_Close"]
        /
        df["SMA_20"]
        - 1
    )

    # ========================================================
    # 5. RSI 14
    # ========================================================

    df["RSI_14"] = calculate_rsi(
        prices=df["Adj_Close"],
        window=14,
    )

    # ========================================================
    # 6. ZWROT W TRAKCIE SESJI
    # ========================================================

    df["Intraday_Return"] = (
        df["Close"]
        /
        df["Open"]
        - 1
    )

    # ========================================================
    # 7. ZAKRES RUCHU PODCZAS SESJI
    # ========================================================

    df["Daily_Range"] = (
        (
            df["High"]
            - df["Low"]
        )
        /
        df["Close"]
    )

    return df


# ============================================================
# CECHY DLA CAŁEGO DATASETU
# ============================================================

def create_market_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    # --------------------------------------------------------
    # FORMAT DAT
    # --------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    # --------------------------------------------------------
    # FORMAT LICZB
    # --------------------------------------------------------

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj_Close",
        "Volume",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # WALIDACJA
    # --------------------------------------------------------

    validate_market_data(
        df
    )

    # --------------------------------------------------------
    # OBLICZENIA OSOBNO DLA KAŻDEGO TICKERA
    # --------------------------------------------------------

    feature_frames = []

    for ticker, ticker_df in df.groupby(
        "Ticker",
        sort=False,
    ):

        print(
            f"  Liczenie cech dla: {ticker}"
        )

        ticker_features = (
            calculate_ticker_features(
                ticker_df
            )
        )

        feature_frames.append(
            ticker_features
        )

    # --------------------------------------------------------
    # ŁĄCZENIE
    # --------------------------------------------------------

    features_df = pd.concat(
        feature_frames,
        ignore_index=True,
    )

    features_df = features_df.sort_values(
        by=[
            "Ticker",
            "Date",
        ]
    ).reset_index(
        drop=True
    )

    return features_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 80
    )

    print(
        "TWORZENIE CECH RYNKOWYCH"
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

    df_market = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"\nLiczba wierszy wejściowych: "
        f"{len(df_market)}"
    )

    print(
        f"Liczba tickerów: "
        f"{df_market['Ticker'].nunique()}"
    )

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    df_features = create_market_features(
        df_market
    )

    # ========================================================
    # INFORMACJA O GOTOWYCH WIERSZACH
    # ========================================================

    ready_mask = (
        df_features[
            MODEL_FEATURE_COLUMNS
        ]
        .notna()
        .all(
            axis=1
        )
    )

    ready_count = int(
        ready_mask.sum()
    )

    # ========================================================
    # ZAPIS
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df_features.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # PODSUMOWANIE
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "PODSUMOWANIE"
    )

    print(
        "=" * 80
    )

    print(
        f"\nLiczba wszystkich wierszy: "
        f"{len(df_features)}"
    )

    print(
        f"Liczba wierszy z kompletem "
        f"cech modelowych: {ready_count}"
    )

    print(
        "\nBraki w cechach:"
    )

    print(
        df_features[
            MODEL_FEATURE_COLUMNS
        ]
        .isna()
        .sum()
    )

    # ========================================================
    # PODGLĄD OSTATNICH WIERSZY DLA KAŻDEGO TICKERA
    # ========================================================

    columns_to_show = [
        "Ticker",
        "Date",
        "Adj_Close",
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

    preview = (
        df_features.groupby(
            "Ticker",
            group_keys=False,
        )
        .tail(3)
    )

    print(
        "\nOstatnie 3 sesje każdego tickera:"
    )

    print(
        preview[
            columns_to_show
        ].to_string(
            index=False
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