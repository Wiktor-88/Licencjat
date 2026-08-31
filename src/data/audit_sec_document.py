# Plik dodatkowy - sprawdzenie typów plików z raportów SEC

import json
import logging
import re

from collections import Counter, defaultdict
from pathlib import Path


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "sec-edgar-filings"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "audit_summary.json"

TYPE_PATTERN = re.compile(r"<TYPE>\s*([^\r\n<]+)",
                          flags=re.IGNORECASE)


def extract_document_types(file_path: Path) -> list[str]:
    """
    Zwraca wszystkie typy dokumentów oznaczone <TYPE> znalezione w plikach full-submission.txt
    """

    document_types = []

    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as file:

            for line in file:
                match = TYPE_PATTERN.search(line)

                if match:
                    document_type = match.group(1).strip().upper()
                    document_types.append(document_type)

    except Exception:
        logger.exception("Błąd odczytu pliku %s", file_path)

    return document_types


def audit_sec_documents(base_input_dir: Path, tickers: list[str], output_file: Path) -> None:
    """
    Spsiuje typy dokumentów w 8-K i zapisuje do pliku JSON
    """

    global_counter = Counter()
    types_by_ticker = defaultdict(Counter)

    for ticker in tickers:
        ticker_path = base_input_dir / ticker / "8-K"

        if not ticker_path.exists():
            logger.warning("Brak folderu dla %s", ticker)
            continue

        file_paths = sorted(ticker_path.rglob("full-submission.txt"))

        logger.info("Audyt %s,  liczba filingów: %d", ticker, len(file_paths))

        for file_path in file_paths:
            document_types = extract_document_types(file_path)

            global_counter.update(document_types)
            types_by_ticker[ticker].update(document_types)

    summary_data = {
        "global_summary": dict(global_counter),
        "ticker_summary": {ticker: dict(counts) for ticker, counts in types_by_ticker.items()}
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            summary_data,
            file,
            indent=4,
            ensure_ascii=False,
        )

    logger.info("Koniec, wyniki są w %s", output_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    target_tickers = [
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

    audit_sec_documents(base_input_dir=INPUT_DIRECTORY, tickers=target_tickers, output_file=OUTPUT_FILE)