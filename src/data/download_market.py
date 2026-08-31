# Drugi plik - pobieranie danych finanosywch z Yahoo Finance (yfinance)


import logging
import pandas as pd
import yfinance as yf
from pathlib import Path


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "market_data.csv"


def download_market_data(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Pobiera dane z yfinanse z zadanej daty dla danych tickerów i
    zapiuje je w ramce danych
    """

    logger.info("Pobieranie danych dla %d tickerów, zakres: %s -> %s",
        len(tickers),
        start_date,
        end_date)

    # Pobieranie danych finanowych co jeden dzień (interval = 1d),
    # blokuje wsteczną korekte cen, surowe ceny (auto_adjust = False)
    # tylko dane z godzin handlu (prepost = False)
    # Brak dodatkowych kolumn informujących o dniach i wartościach wypłat dywidend oraz podziałów akcji (actions = False)
    # progres = false, brak paska informującego
    # Hierarchiczny indeks kolumnowy
    df = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        prepost=False,
        actions=False,
        progress=False,
        multi_level_index=True,
    )

    if df.empty:
        raise ValueError('Brak danych z Yfinanse')

    # Zmiana na zwykłą ramke danych
    df = df.stack(level=1, future_stack=True).rename_axis(["Date", "Ticker"]).reset_index()

    df = df.rename(columns={"Adj Close": "Adj_Close"})

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

    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Brakuje kolumn: {missing_columns}")

    # Uporządkowanie aby obserwacje były najpierw tych samych tikerów, potem po dacie
    df = df[required_columns].sort_values(["Ticker", "Date"]).reset_index(drop=True)


    # Sprawdznie czy nie ma tej samej obserwacji wiecej niz raz
    duplicated_rows = df.duplicated(subset=["Ticker", "Date"]).sum()

    if duplicated_rows > 0:
        raise ValueError(f'Zduplikowane pary Ticker-Date: {duplicated_rows}')


    # Sprawdzenie czy wszystkie tikery sie pobrały (działanie na zbiorach matematyzcnych)
    downloaded_tickers = set(df["Ticker"].unique())

    missing_tickers = set(tickers)- downloaded_tickers
    
    if missing_tickers:
        raise ValueError(f"Brak danych dla tickerów: {sorted(missing_tickers)}")


    logger.info("Pobieranie zakończone, zakres danych: %s -> %s",
        df["Date"].min().date(),
        df["Date"].max().date())

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    tickers = [
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "GOOGL",
        "META",
        "TSLA",
        "AMD",
        "INTC",
        "AVGO",
        "NFLX",
        "ADBE",
        "QCOM",
        "QQQ",
    ]

    start_date = "2019-06-01"
    end_date = "2026-08-17"

    market_data = download_market_data(tickers=tickers, start_date=start_date,end_date=end_date)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    market_data.to_csv(OUTPUT_FILE, index=False)

    logger.info("Liczba wszystkich wierszy: %d", len(market_data))

    logger.info("Liczba obserwacji per ticker:\n%s",
                market_data["Ticker"].value_counts().sort_index().to_string())

    logger.info("Dane zapisane w %s", OUTPUT_FILE)