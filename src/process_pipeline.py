# Uruchomienie całego pipelineu projektu w ustalonej kolejności

import argparse
import logging
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.features.clean_text import clean_html_document


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "sec-edgar-filings"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "cleaned_texts"
METADATA_OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "sec_document_metadata.csv"

TARGET_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "INTC", "AVGO", "NFLX", "ADBE", "QCOM"]

PIPELINE_STAGES = [
    ("audit-sec", "src.data.audit_sec_document"),
    ("clean-sec", None),
    ("sentiment-blocks", "src.features.sentiment"),
    ("sentiment-features", "src.features.aggregate_sentiment"),
    ("align-events", "src.features.align_sec_events"),
    ("market-features", "src.features.market_features"),
    ("model-dataset", "src.features.build_model_dataset"),
    ("audit-dataset", "src.features.audit_model_dataset"),
    ("sequence-dataset", "src.features.build_sequence_dataset"),
    ("methodology-tests", "src.test_methodology"),
    ("logistic", "src.models.train_baseline"),
    ("tree", "src.models.train_decision_tree"),
    ("xgboost", "src.models.train_xgboost"),
    ("catboost", "src.models.train_catboost"),
    ("tabnet", "src.models.train_tabnet"),
    ("lstm", "src.models.train_lstm"),
    ("transformer", "src.models.train_transformer"),
    ("test-null", "src.models.metrics_permutations_test"),
    ("test-sec", "src.models.A_vs_B_test"),
    ("test-finbert", "src.models.B_vs_C_test"),
    ("financial", "src.models.evaluate_financial_metrics"),
    ("xai-logistic", "src.xai.run_logistic"),
    ("xai-tree", "src.xai.run_tree"),
    ("xai-xgboost", "src.xai.run_xgboost"),
    ("xai-catboost", "src.xai.run_catboost"),
    ("xai-tabnet", "src.xai.run_tabnet"),
    ("xai-lstm", "src.xai.run_lstm"),
    ("xai-transformer", "src.xai.run_transformer")]

DOWNLOAD_STAGES = [("download-sec", "src.data.download_sec"), ("download-market", "src.data.download_market")]


def extract_sec_metadata(file_path: Path) -> tuple[str, str]:
    """Pobiera datę złożenia i dokładny czas akceptacji raportu SEC."""
    raw_content = file_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"<ACCEPTANCE-DATETIME>\s*(\d{14})", raw_content, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Nie znaleziono ACCEPTANCE-DATETIME w pliku: {file_path}")

    acceptance = datetime.strptime(match.group(1), "%Y%m%d%H%M%S").replace(tzinfo=ZoneInfo("America/New_York"))
    return acceptance.date().isoformat(), acceptance.isoformat()


def process_all_reports(base_input_dir: Path, output_dir: Path, tickers: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows = []

    for ticker in tickers:
        file_paths = sorted((base_input_dir / ticker / "8-K").rglob("full-submission.txt"))
        logger.info("%s: znaleziono %d raportów.", ticker, len(file_paths))

        for file_path in file_paths:
            filing_date, acceptance_datetime_et = extract_sec_metadata(file_path)
            accession = file_path.parent.name
            cleaned_documents = clean_html_document(file_path)
            if not cleaned_documents:
                logger.warning("Brak dokumentów dla: %s", accession)
                continue

            for document in cleaned_documents:
                source_type, clean_text = document["source_type"], document["text"]
                if not clean_text.strip():
                    continue

                output_filename = f"{ticker}_{filing_date}_{accession}_{source_type.replace('.', '-')}.txt"
                (output_dir / output_filename).write_text(clean_text, encoding="utf-8")
                metadata_rows.append({"Ticker": ticker, "Filing_Date": filing_date,
                    "Acceptance_DateTime_ET": acceptance_datetime_et, "Accession": accession,
                    "Source_Type": source_type, "Cleaned_File": output_filename})

    columns = ["Ticker", "Filing_Date", "Acceptance_DateTime_ET", "Accession", "Source_Type", "Cleaned_File"]
    metadata = pd.DataFrame(metadata_rows, columns=columns)
    metadata = metadata.sort_values(["Ticker", "Filing_Date", "Accession", "Source_Type"]).reset_index(drop=True)
    expected_files = set(metadata["Cleaned_File"])
    stale_files = [path for ticker in tickers for path in output_dir.glob(f"{ticker}_*.txt") if path.name not in expected_files]
    for path in stale_files:
        path.unlink()
    if stale_files:
        logger.info("Usunięto %d nieaktualnych plików z katalogu cleaned_texts.", len(stale_files))

    METADATA_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(METADATA_OUTPUT_FILE, index=False)
    logger.info("Zapisano %d dokumentów do %s", len(metadata), METADATA_OUTPUT_FILE)


def select_stages(from_stage: str | None, to_stage: str | None, skip_xai: bool) -> list[tuple[str, str | None]]:
    stages = PIPELINE_STAGES.copy()
    names = [name for name, _ in stages]
    start = names.index(from_stage) if from_stage else 0
    end = names.index(to_stage) + 1 if to_stage else len(stages)
    if start >= end:
        raise ValueError("Etap początkowy musi występować przed etapem końcowym.")

    selected = stages[start:end]
    return [(name, module) for name, module in selected if not (skip_xai and name.startswith("xai-"))]


def run_module(module: str, dry_run: bool) -> None:
    command = [sys.executable, "-m", module]
    if dry_run:
        logger.info("DRY RUN | %s", " ".join(command))
        return

    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run_pipeline(from_stage: str | None, to_stage: str | None, skip_xai: bool, dry_run: bool,
    download_data: bool) -> None:
    stages = select_stages(from_stage, to_stage, skip_xai)
    if download_data:
        for name, module in DOWNLOAD_STAGES:
            logger.info("Start: %s", name)
            run_module(module, dry_run)
    logger.info("Etapy pipeline'u: %s", ", ".join(name for name, _ in stages))

    for number, (name, module) in enumerate(stages, start=1):
        logger.info("[%d/%d] Start: %s", number, len(stages), name)
        if module:
            run_module(module, dry_run)
        elif dry_run:
            logger.info("DRY RUN | czyszczenie raportów SEC")
        else:
            process_all_reports(INPUT_DIRECTORY, OUTPUT_DIRECTORY, TARGET_TICKERS)
        logger.info("[%d/%d] Koniec: %s", number, len(stages), name)


def parse_args() -> argparse.Namespace:
    stage_names = [name for name, _ in PIPELINE_STAGES]
    parser = argparse.ArgumentParser(description="Pełny pipeline danych, modeli, testów i XAI.")
    parser.add_argument("--from-stage", choices=stage_names, help="Etap, od którego wznowić obliczenia.")
    parser.add_argument("--to-stage", choices=stage_names, help="Ostatni etap do uruchomienia.")
    parser.add_argument("--skip-xai", action="store_true", help="Pomija kosztowne obliczenia XAI.")
    parser.add_argument("--download-data", action="store_true", help="Przed obliczeniami aktualizuje dane SEC i rynkowe.")
    parser.add_argument("--dry-run", action="store_true", help="Pokazuje kolejność bez wykonywania obliczeń.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.from_stage, args.to_stage, args.skip_xai, args.dry_run, args.download_data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    main()
