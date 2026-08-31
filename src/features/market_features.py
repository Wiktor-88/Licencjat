# Plik szósty - słuzy on głównie do tworzenia cech ML z dostępnych rzeczy

from pathlib import Path

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "market_data.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "market_features.csv"

IDENTIFIER_COLUMNS = [
    "Ticker",
    "Date",
]

PRICE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj_Close",
]

NUMERIC_COLUMNS = PRICE_COLUMNS + ["Volume"]
REQUIRED_COLUMNS = IDENTIFIER_COLUMNS + NUMERIC_COLUMNS


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


# Walidacja
def validate_market_data(df: pd.DataFrame) -> None:
    """Sprawdza poprawność danych rynkowych przed tworzeniem cech"""

    if df.empty:
        raise ValueError('market_data.csv nie zawiera żadnych rekordów')
    

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing_columns:
        raise ValueError("Brakuje wymaganych kolumn w market_data.csv:\n \n".join(missing_columns))


    missing_identifiers = df[IDENTIFIER_COLUMNS].isna().any(axis=1)

    if missing_identifiers.any():
        invalid_rows = df.loc[missing_identifiers, IDENTIFIER_COLUMNS].head(10)
        raise ValueError(f"Znaleziono brakujące wartości Ticker lub Date Liczba rekordów: "
                        f'{missing_identifiers.sum()}{invalid_rows.to_string(index=False)}')


   
    duplicate_mask = df.duplicated(subset=["Ticker", "Date"], keep=False)

    if duplicate_mask.any():
        duplicates = df.loc[duplicate_mask, ["Ticker", "Date"]]
        raise ValueError(f"Znaleziono duplikaty Ticker + Date:\n {duplicates}")


    non_numeric_columns = [column for column in NUMERIC_COLUMNS if not pd.api.types.is_numeric_dtype(df[column])]

    if non_numeric_columns:
        raise ValueError("Kolumny powinny zawierać dane numeryczne:" + "\n".join(non_numeric_columns))
    

    missing_counts = df[NUMERIC_COLUMNS].isna().sum()

    missing_counts = missing_counts[missing_counts > 0]

    if not missing_counts.empty:
        raise ValueError(f"Znaleziono braki w danych OHLCV:\n {missing_counts}")


    # Nieskończone wartości
    finite_mask = np.isfinite(df[NUMERIC_COLUMNS]).all(axis=1)

    if not finite_mask.all():
        invalid_rows = df.loc[~finite_mask, IDENTIFIER_COLUMNS + NUMERIC_COLUMNS].head(10)

        raise ValueError("Znaleziono wartości inf lub -inf w danych"
                        f"Liczba rekordów: {(~finite_mask).sum()}\n {invalid_rows.to_string(index=False)}")


    # Ceny musza byc dodatnie
    price_columns = ["Open", "High", "Low", "Close", "Adj_Close"]

    invalid_prices = (df[price_columns] <= 0).any(axis=1)

    if invalid_prices.any():

        invalid_rows = df.loc[invalid_prices, ["Ticker","Date"] + price_columns]
        raise ValueError(f"Znaleziono ujemne ceny: {invalid_rows}")

    
    # Ujemny wolumen
    invalid_volume = df["Volume"] < 0

    if invalid_volume.any():
        invalid_rows = df.loc[invalid_volume, IDENTIFIER_COLUMNS + ["Volume"]].head(10)
        raise ValueError(f"Znaleziono ujemny wolumen {invalid_rows.to_string(index=False)}")


    # Spójność OHCL
    invalid_ohlc = (
        (df["High"] < df["Low"])
        | (df["Open"] > df["High"])
        | (df["Open"] < df["Low"])
        | (df["Close"] > df["High"])
        | (df["Close"] < df["Low"])
    )

    if invalid_ohlc.any():
        invalid_rows = df.loc[invalid_ohlc, IDENTIFIER_COLUMNS + ["Open","High","Low","Close"]].head(10)

        raise ValueError(f"Znaleziono niespójne wartości OHLC {invalid_rows.to_string(index=False)}")


# RSI
def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Oblicza RSI metodą wygładzania Wildera"""

    if window <= 0:
        raise ValueError('Okno RSI musi być większe od 0')

    delta = prices.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = pd.Series(np.nan, index=prices.index, dtype=float)

    average_loss = pd.Series(np.nan, index=prices.index, dtype=float,)

    if len(prices) <= window:
        return average_gain

    # Pierwsza wartość zgodnie z metodą Wildera:
    # średnia z pierwszych window zmian cen
    average_gain.iloc[window] = gains.iloc[1 : window + 1].mean()
    average_loss.iloc[window] = losses.iloc[1 : window + 1].mean()
    

    # Kolejne wartości są wygładzane metodą Wildera
    for i in range(window + 1, len(prices)):
        average_gain.iloc[i] = ((average_gain.iloc[i - 1] * (window - 1))+ gains.iloc[i]) / window

        average_loss.iloc[i] = ((average_loss.iloc[i - 1] * (window - 1)) + losses.iloc[i]) / window

    relative_strength = (average_gain / average_loss)

    rsi = (100 - (100 / (1 + relative_strength)))

    # Brak zmian ceny przez całe okno oznacza neutralne RSI
    neutral_mask = ((average_gain == 0) & (average_loss == 0))

    rsi = rsi.mask(neutral_mask, 50.0)

    return rsi



# Cechy dla jednej firmy (tickera)
def calculate_ticker_features(ticker_df: pd.DataFrame) -> pd.DataFrame:

    df = ticker_df.copy()

    df = df.sort_values(by="Date").reset_index(drop=True)

    # Logarytmiczne stopy zwrotu
    df["Log_Return_1D"] = np.log(df["Adj_Close"] / df["Adj_Close"].shift(1))
    df["Log_Return_3D"] = np.log(df["Adj_Close"] / df["Adj_Close"].shift(3))
    df["Log_Return_5D"] = np.log(df["Adj_Close"] / df["Adj_Close"].shift(5))

    # Zmienność 14 - dniowa
    df["Volatility_14D"] = df["Log_Return_1D"].rolling(window=14, min_periods=14).std()
    

    # Średni wolumen z 20 ostatnich sesji
    df["Avg_Volume_20D_Previous"] = df["Volume"].shift(1).rolling(window=20, min_periods=20).mean()
    df["Relative_Volume_20D"] = df["Volume"] / df["Avg_Volume_20D_Previous"]
    

    # SMA 20
    df["SMA_20"] = df["Adj_Close"].rolling(window=20, min_periods=20).mean()
    df["Price_to_SMA20"] = df["Adj_Close"] / df["SMA_20"] - 1
    

    # RSi 14
    df["RSI_14"] = calculate_rsi(prices=df["Adj_Close"], window=14,)


    # Zwrto w trkacie sesji
    df["Intraday_Return"] = df["Close"] / df["Open"] - 1
    

    # Zakres ruchu podczas sesji
    df["Daily_Range"] = (df["High"] - df["Low"]) / df["Close"]

    return df



# Cechy dla całego datasetu
def create_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tworzy cechy rynkowe dla wszystkich firm"""

    df = df.copy()

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing_columns:
        raise ValueError("Brakuje wymaganych kolumn :\n" + "\n".join(missing_columns))


    # Format danych 
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

   
    validate_market_data(df)


    # Cechy dla każdego tickera
    feature_frames = []

    for ticker, ticker_df in df.groupby("Ticker", sort=False):
        logger.info("Liczenie cech dla: %s", ticker)
        feature_frames.append(calculate_ticker_features(ticker_df))

    
    # Ostateczne łączenie
    features_df = pd.concat(feature_frames, ignore_index=True)

    return features_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    


# Cechy rolling Z-60
def add_rolling_zscore_features(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Dodaje rolling Z-score względem poprzednich sesji"""

    if window < 2:
        raise ValueError("Okno Z-score musi być co najmniej 2")

    df = df.copy().sort_values(["Ticker", "Date"])

    zscore_source_columns = [
        "Log_Return_1D",
        "Log_Return_3D",
        "Log_Return_5D",
        "Volatility_14D",
        "Relative_Volume_20D",
        "Price_to_SMA20",
        "Intraday_Return",
        "Daily_Range",
    ]

    for column in zscore_source_columns:
        grouped = df.groupby("Ticker")[column]

        rolling_mean = grouped.transform(lambda s: s.shift(1).rolling(window, min_periods=window).mean())

        rolling_std = grouped.transform(lambda s: s.shift(1).rolling(window, min_periods=window).std())

        zscore_column = f"{column}_Z{window}"

        df[zscore_column] = (df[column] - rolling_mean) / rolling_std.replace(0, np.nan)

    return df



def main() -> None:

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku:\n{INPUT_FILE}")


    print("Robienie cech rynkowych")
    df_market = pd.read_csv(INPUT_FILE)

    print(f"\nLiczba wierszy: {len(df_market)}")

    if "Ticker" in df_market.columns:
        print(f"Liczba tickerów: {df_market['Ticker'].nunique()}")


    
    df_features = create_market_features(df_market)
    df_features = add_rolling_zscore_features(df_features, window=60)

    
    ready_mask = df_features[MODEL_FEATURE_COLUMNS].notna().all(axis=1)
    ready_count = int(ready_mask.sum())

    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(OUTPUT_FILE, index=False)

    
    print("\n" + "-" * 80)
    print("Podsuwmoanie")

    print(f"\nLiczba wierszy: {len(df_features)}")
    print(f"Liczba wierszy z kompletem cech modelowych: {ready_count}")

    print("\nBraki w cechach:")
    print(df_features[MODEL_FEATURE_COLUMNS].isna().sum())

    columns_to_show = [
        "Ticker", "Date", "Adj_Close",
        "Log_Return_1D", "Log_Return_3D", "Log_Return_5D",
        "Volatility_14D", "Relative_Volume_20D", "RSI_14",
        "Price_to_SMA20", "Intraday_Return", "Daily_Range",
    ]

    preview = df_features.groupby("Ticker", group_keys=False).tail(3)

    print("\nOstatnie 3 sesje każdego tickera:")
    print(preview[columns_to_show].to_string(index=False))



if __name__ == "__main__":
    main()