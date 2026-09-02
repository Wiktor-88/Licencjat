# Plik zawierjący elemty wspólne dla całej część XAI

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
XAI_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "xai"

TARGET_COLUMN = "Target_Tradable_Abnormal_1D"

IDENTIFIER_COLUMNS = ["Ticker",
                      "Accession",
                      "Filing_Date",
                      "Feature_Cutoff_Session",
                      "Event_Session",
                      "Publication_Period",
                      "Test_Year"]


def get_model_xai_dir(model_name: str) -> Path:
    if not model_name.strip():
        raise ValueError("model_name nie może być pusty")

    output_dir = XAI_OUTPUT_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_local_xai_dir(model_name: str, accession: str) -> Path:
    if not str(accession).strip():
        raise ValueError("Accession nie może być pusty")

    output_dir = get_model_xai_dir(model_name) / "local" / str(accession)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def validate_xai_dataframe(df: pd.DataFrame) -> None:
    required = {"Ticker", "Accession", TARGET_COLUMN}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Brakuje kolumn wymaganych przez XAI: {sorted(missing)}")
    if df.empty:
        raise ValueError("DataFrame przekazany do XAI jest pusty")

    duplicates = df.duplicated(["Ticker", "Accession"], keep=False)
    if duplicates.any():
        rows = df.loc[duplicates, ["Ticker", "Accession"]]
        raise ValueError("Znaleziono zduplikowane filingi:\n" + rows.to_string(index=False))


def get_event_by_accession(df: pd.DataFrame, accession: str) -> pd.Series:
    validate_xai_dataframe(df)

    matches = df[df["Accession"].astype(str) == str(accession)]

    if matches.empty:
        raise ValueError(f"Nie znaleziono filingu Accession={accession}")
    if len(matches) > 1:
        raise ValueError(f"Accession nie identyfikuje jednoznacznie obserwacji: {accession}")

    return matches.iloc[0]


def select_local_classification_examples(df: pd.DataFrame,
                                        probability_column: str,
                                        prediction_column: str) -> pd.DataFrame:
    validate_xai_dataframe(df)

    missing = {probability_column, prediction_column} - set(df.columns)

    if missing:
        raise ValueError(f"Brak kolumn do local XAI: {sorted(missing)}")

    work = df.copy()
    work[probability_column] = pd.to_numeric(work[probability_column], errors="coerce")
    work[prediction_column] = pd.to_numeric(work[prediction_column], errors="coerce")
    work[TARGET_COLUMN] = pd.to_numeric(work[TARGET_COLUMN], errors="coerce")

    work = work[np.isfinite(work[probability_column])
                & work[prediction_column].isin([0, 1])
                & work[TARGET_COLUMN].isin([0, 1])].copy()

    cases = [
        ("true_positive", (work[TARGET_COLUMN] == 1) & (work[prediction_column] == 1), "max"),
        ("true_negative", (work[TARGET_COLUMN] == 0) & (work[prediction_column] == 0), "min"),
        ("false_positive", (work[TARGET_COLUMN] == 0) & (work[prediction_column] == 1), "max"),
        ("false_negative", (work[TARGET_COLUMN] == 1) & (work[prediction_column] == 0), "min"),
    ]

    selected = []

    for example_type, mask, direction in cases:
        candidates = work.loc[mask]

        if candidates.empty:
            logger.warning("Brak przykładu typu %s", example_type)
            continue

        index = (candidates[probability_column].idxmax() if direction == "max"
            else candidates[probability_column].idxmin())

        row = work.loc[index].copy()
        row["XAI_Example_Type"] = example_type
        selected.append(row)

    if not selected:
        raise ValueError("Nie znaleziono obserwacji do local XAI")

    return pd.DataFrame(selected).reset_index(drop=True)


def build_event_metadata(row: pd.Series) -> dict[str, Any]:
    metadata = {}

    for column in IDENTIFIER_COLUMNS:
        if column not in row.index:
            continue

        value = row[column]
        if pd.isna(value):
            metadata[column] = None
        elif isinstance(value, pd.Timestamp):
            metadata[column] = value.isoformat()
        else:
            metadata[column] = value

    if TARGET_COLUMN in row.index:
        value = row[TARGET_COLUMN]
        metadata[TARGET_COLUMN] = None if pd.isna(value) else int(value)

    return metadata


def save_json(data: dict[str, Any], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(data,
                  file,
                  ensure_ascii=False,
                  indent=2,
                  default=_json_serializer)

    logger.info("Zapisano JSON: %s", output_file)


def _json_serializer(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None

    raise TypeError(f"Nieobsługiwany typ JSON: {type(value)}")


def save_dataframe(df: pd.DataFrame, output_file: Path) -> None:
    if df.empty:
        raise ValueError(f"Nie można zapisać pustego DataFrame: {output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    logger.info("Zapisano tabelę XAI: %s", output_file)
