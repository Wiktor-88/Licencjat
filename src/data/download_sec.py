import os                                       # Moduł służacy do interakcji z systemem plików i ścieżkami

from sec_edgar_downloader import Downloader     # Moduł do pobierania dokumentów z bazy SEC EDGAR
from dotenv import load_dotenv                  # Moduł do wczytywania zmiennych środowiskowych z pliku .env

load_dotenv()                                   # Wczytuje zmienne środowiskowe z pliku .env

def fetch_8k_reports(
    tickers: list,
    start_date: str,
    end_date: str,
) -> str:
    """
    Pobiera raporty 8-K dla podanej listy tickerów
    z określonego zakresu dat.
    """

    company_name = os.getenv("SEC_COMPANY_NAME")
    email = os.getenv("SEC_EMAIL")

    if not company_name or not email:
        raise ValueError(
            "Błąd: Zmienne środowiskowe "
            "nie zostały załadowane prawidłowo."
        )

    download_root = os.path.abspath(
        os.path.join("data", "raw")
    )

    dl = Downloader(
        company_name,
        email,
        download_root,
    )

    for ticker in tickers:
        print(
            f"Pobieram raporty 8-K dla spółki "
            f"{ticker}..."
        )

        downloaded_count = dl.get(
            "8-K",
            ticker,
            after=start_date,
            before=end_date,
            include_amends=False,
        )

        print(
            f"  Pobrano filingów: "
            f"{downloaded_count}"
        )

    download_path = os.path.join(
        download_root,
        "sec-edgar-filings",
    )

    print(
        "Gotowe! Pliki zostały zapisane "
        f"w folderze: {download_path}"
    )

    return download_path

if __name__ == "__main__":
    my_tickers = [
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

    fetch_8k_reports(
        tickers=my_tickers,
        start_date="2020-01-01",
        end_date="2026-08-13",
    )