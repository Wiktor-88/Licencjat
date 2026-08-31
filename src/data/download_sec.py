# Jest to pierwszy plik - pobieranie danych z SEC EDGAR


import logging
import os

from pathlib import Path
from dotenv import load_dotenv
from sec_edgar_downloader import Downloader


load_dotenv()

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"


def fetch_8k_reports(tickers: list[str], start_date: str, end_date: str) -> Path:
    """
    Funkcja ta pobiera raporty z wybranego zakresu dat
    dla podanych tickerów (firm),
    wynik jest zapisywany
    """

    company_name = os.getenv("SEC_COMPANY_NAME")
    email = os.getenv("SEC_EMAIL")

    if not company_name or not email:
        raise ValueError("Brak wymaganych zmiennych srodowiskowych SEC")

    download_root = DATA_RAW_DIR.resolve()

    # Obiekt do pobierania do danej ścieżki
    downloader = Downloader(company_name, email, download_root)

    # Faktyczne pobieranaie
    for ticker in tickers:
        logger.info("Pobieranie 8-K dla %s", ticker)

        try:
            # Pobieramy komunikaty nie zmienione, czyli inclund_amends = False
            count = downloader.get(
                "8-K",
                ticker,
                after=start_date,
                before=end_date,
                include_amends=False,
            )

            logger.info("Zakończono pobieranie dla %s, liczba raportów: %s", ticker, count)

        except Exception:
            logger.exception("Błąd pobierania dla %s", ticker)

    return download_root / "sec-edgar-filings"


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
    ]

    fetch_8k_reports(tickers=tickers, start_date="2020-01-01", end_date="2026-08-13")