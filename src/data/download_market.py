from pathlib import Path

import pandas as pd
import yfinance as yf

# ============================================================
# ŚCIEŻKI
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_FILE = (PROJECT_ROOT / "data" / "processed" / "market_data.csv")


# ============================================================
# KONFIGURACJA
# ============================================================

TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    'QQQ'
]

START_DATE = "2019-10-01"

# yfinance traktuje end jako datę WYŁĄCZNĄ.
# Czyli end="2026-08-17" pobiera dane maksymalnie
# do 2026-08-16.
END_DATE = "2026-08-17"


# ============================================================
# POBRANIE DANYCH JEDNEJ SPÓŁKI
# ============================================================

def download_ticker_data(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:

    print(
        f"\nPobieranie danych dla: {ticker}"
    )

    df = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        interval="1d",

        # Chcemy zachować zarówno Close,
        # jak i Adj Close.
        auto_adjust=False,

        # Bez danych pre-market / after-hours.
        prepost=False,

        # Nie potrzebujemy dywidend i splitów
        # jako osobnych kolumn na tym etapie.
        actions=False,

        # Czytelniejszy output w terminalu.
        progress=False,

        # Dla jednego tickera chcemy zwykłe
        # kolumny zamiast MultiIndex.
        multi_level_index=False,
    )

    # ========================================================
    # WALIDACJA
    # ========================================================

    if df.empty:
        print(
            f"Brak danych dla {ticker}."
        )

        return pd.DataFrame()

    # ========================================================
    # INDEX DATE -> ZWYKŁA KOLUMNA
    # ========================================================

    df = df.reset_index()

    # ========================================================
    # UJEDNOLICENIE NAZW KOLUMN
    # ========================================================

    df = df.rename(
        columns={
            "Adj Close": "Adj_Close",
        }
    )

    # ========================================================
    # DODAJ TICKER
    # ========================================================

    df.insert(
        0,
        "Ticker",
        ticker,
    )

    # ========================================================
    # WYBIERAMY TYLKO POTRZEBNE KOLUMNY
    # ========================================================

    required_columns = [
        "Ticker",
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj_Close",
        "Volume",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{ticker}: brakuje kolumn: "
            f"{missing_columns}"
        )

    df = df[
        required_columns
    ].copy()

    # ========================================================
    # FORMAT DATY
    # ========================================================

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    # ========================================================
    # SORTOWANIE
    # ========================================================

    df = df.sort_values(
        by="Date"
    ).reset_index(
        drop=True
    )

    print(
        f"  Pobrano wierszy: {len(df)}"
    )

    print(
        f"  Zakres: "
        f"{df['Date'].min().date()} "
        f"-> "
        f"{df['Date'].max().date()}"
    )

    return df


# ============================================================
# POBRANIE WSZYSTKICH SPÓŁEK
# ============================================================

def download_market_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:

    all_data = []

    for ticker in tickers:

        ticker_df = download_ticker_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )

        if ticker_df.empty:
            continue

        all_data.append(
            ticker_df
        )

    if not all_data:
        raise ValueError(
            "Nie udało się pobrać danych "
            "dla żadnego tickera."
        )

    # ========================================================
    # ŁĄCZENIE
    # ========================================================

    market_df = pd.concat(
        all_data,
        ignore_index=True,
    )

    # ========================================================
    # SORTOWANIE
    # ========================================================

    market_df = market_df.sort_values(
        by=[
            "Ticker",
            "Date",
        ]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # KONTROLA DUPLIKATÓW
    # ========================================================

    duplicate_mask = (
        market_df.duplicated(
            subset=[
                "Ticker",
                "Date",
            ],
            keep=False,
        )
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"Znaleziono {duplicate_count} "
            f"zduplikowanych wierszy "
            f"Ticker + Date."
        )

    return market_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 70
    )

    print(
        "POBIERANIE DANYCH RYNKOWYCH"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTickery: {TICKERS}"
    )

    print(
        f"Zakres: {START_DATE} -> {END_DATE}"
    )

    # ========================================================
    # DOWNLOAD
    # ========================================================

    df_market = download_market_data(
        tickers=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    # ========================================================
    # ZAPIS
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df_market.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # PODSUMOWANIE
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PODSUMOWANIE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nLiczba wszystkich wierszy: "
        f"{len(df_market)}"
    )

    print(
        "\nLiczba obserwacji per ticker:"
    )

    print(
        df_market[
            "Ticker"
        ].value_counts()
    )

    print(
        "\nPierwsze wiersze:"
    )

    print(
        df_market.head(
            10
        ).to_string(
            index=False
        )
    )

    print(
        f"\nWyniki zapisano do:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        "=" * 70
    )