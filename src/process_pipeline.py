# Pipeline odpowiedzialny za wszystko

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import logging
import re

import pandas as pd

from features.clean_text import clean_html_document


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "sec-edgar-filings"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "cleaned_texts"
METADATA_OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "sec_document_metadata.csv"

TARGET_TICKERS = [
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


# Metadane
def extract_sec_metadata(file_path: Path) -> tuple[str, str]:
    """Pobiera Filing_Date i Acceptance_DateTime_ET z raportu SEC."""

    raw_content = file_path.read_text(encoding="utf-8", errors="ignore")

    match = re.search(r"<ACCEPTANCE-DATETIME>\s*(\d{14})", raw_content, flags=re.IGNORECASE)

    if not match:
        raise ValueError(f"Nie znaleziono ACCEPTANCE-DATETIME w pliku: {file_path}")

    acceptance_datetime = datetime.strptime(match.group(1), "%Y%m%d%H%M%S",).replace(tzinfo=ZoneInfo("America/New_York"))

    return ( acceptance_datetime.date().isoformat(), acceptance_datetime.isoformat())


# Przetwarzanie raportó
def process_all_reports(base_input_dir: Path, output_dir: Path, tickers: list[str]) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows = []

    for ticker in tickers:
        file_paths = sorted((base_input_dir / ticker / "8-K").rglob("full-submission.txt"))

        logger.info("%s: znaleziono %d raportów.", ticker, len(file_paths),)

        for file_path in file_paths:
            filing_date, acceptance_datetime_et = extract_sec_metadata(file_path)
            accession = file_path.parent.name

            cleaned_documents = clean_html_document(str(file_path))

            if not cleaned_documents:
                logger.warning("Brak dokumentów dla: %s", accession)
                continue

            for document in cleaned_documents:
                source_type = document["source_type"]
                clean_text = document["text"]

                if not clean_text.strip():
                    continue

                safe_source_type = source_type.replace(".", "-")

                output_filename = f"{ticker}_{filing_date}_{accession}_{safe_source_type}.txt"

                (output_dir / output_filename).write_text(clean_text, encoding="utf-8")

                metadata_rows.append({
                    "Ticker": ticker,
                    "Filing_Date": filing_date,
                    "Acceptance_DateTime_ET": acceptance_datetime_et,
                    "Accession": accession,
                    "Source_Type": source_type,
                    "Cleaned_File": output_filename,
                })

    metadata_df = pd.DataFrame(
        metadata_rows,
        columns=[
            "Ticker",
            "Filing_Date",
            "Acceptance_DateTime_ET",
            "Accession",
            "Source_Type",
            "Cleaned_File"
        ]
    )

    metadata_df.sort_values(
        ["Ticker", "Filing_Date", "Accession", "Source_Type"],
        inplace=True,
        ignore_index=True,
    )

    METADATA_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    metadata_df.to_csv(METADATA_OUTPUT_FILE, index=False)

    logger.info("Zapisano %d dokumentów do %s", len(metadata_df), METADATA_OUTPUT_FILE)



def main() -> None:
    process_all_reports(INPUT_DIRECTORY, OUTPUT_DIRECTORY, TARGET_TICKERS)


if __name__ == "_main_":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    main()