####################################################
# Jest to podstawowy plik, zapewnia on wspólną infrastrukture
# i wspólny format
####################################################



import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)



PROJECT_ROOT = Path(__file__).resolve().parents[2]

XAI_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "xai"



# WSPÓLNE KOLUMNY IDENTYFIKACYJNE

IDENTIFIER_COLUMNS = [
    "Ticker",
    "Accession",
    "Filing_Date",
    "Feature_Cutoff_Session",
    "Event_Session",
    "Publication_Period",
]

TARGET_COLUMN = "Target_Abnormal_1D"


# KATALOGI XAI
def get_model_xai_dir(model_name: str) -> Path:
    if not model_name.strip():
        raise ValueError("model_name nie może być pusty.")

    output_dir = XAI_OUTPUT_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def get_local_xai_dir(
    model_name: str,
    accession: str,
) -> Path:
    if not str(accession).strip():
        raise ValueError("Accession nie może być pusty.")

    output_dir = (
        get_model_xai_dir(model_name)
        / "local"
        / str(accession)
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir



# WALIDACJA OBSERWACJI
def validate_xai_dataframe(
    df: pd.DataFrame,
) -> None:
    required_columns = [
        "Ticker",
        "Accession",
        TARGET_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Brakuje kolumn wymaganych przez XAI: "
            f"{missing_columns}"
        )

    if df.empty:
        raise ValueError(
            "DataFrame przekazany do XAI jest pusty."
        )

    duplicate_mask = df.duplicated(
        subset=["Ticker", "Accession"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_rows = df.loc[
            duplicate_mask,
            ["Ticker", "Accession"],
        ]

        raise ValueError(
            "Znaleziono zduplikowane filing'i:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )



# WYBÓR OBSERWACJI DO LOCAL XAI
def get_event_by_accession(
    df: pd.DataFrame,
    accession: str,
) -> pd.Series:
    validate_xai_dataframe(df)

    matches = df.loc[
        df["Accession"].astype(str) == str(accession)
    ]

    if matches.empty:
        raise ValueError(
            f"Nie znaleziono filingu Accession={accession}."
        )

    if len(matches) > 1:
        raise ValueError(
            "Accession nie identyfikuje jednoznacznie "
            f"obserwacji: {accession}."
        )

    return matches.iloc[0]


def select_local_examples(
    df: pd.DataFrame,
    probability_column: str,
) -> pd.DataFrame:
    validate_xai_dataframe(df)

    if probability_column not in df.columns:
        raise ValueError(
            f"Brak kolumny {probability_column}."
        )

    work = df.copy()

    probabilities = pd.to_numeric(
        work[probability_column],
        errors="coerce",
    )

    valid_mask = (
        probabilities.notna()
        & np.isfinite(probabilities)
    )

    work = work.loc[valid_mask].copy()
    work[probability_column] = probabilities[valid_mask]

    if work.empty:
        raise ValueError(
            "Brak poprawnych prawdopodobieństw "
            "do wyboru lokalnych przykładów."
        )

    selected_indices = {
        "highest_probability": work[
            probability_column
        ].idxmax(),

        "lowest_probability": work[
            probability_column
        ].idxmin(),

        "closest_to_threshold": (
            work[probability_column] - 0.5
        ).abs().idxmin(),
    }

    selected_rows = []

    for example_type, index in selected_indices.items():
        row = work.loc[index].copy()
        row["XAI_Example_Type"] = example_type
        selected_rows.append(row)

    return pd.DataFrame(selected_rows).reset_index(
        drop=True
    )



# METADATA LOKALNEGO WYJAŚNIENIA
def build_event_metadata(
    row: pd.Series,
) -> dict[str, Any]:
    metadata = {}

    for column in IDENTIFIER_COLUMNS:
        if column in row.index:
            value = row[column]

            if pd.isna(value):
                metadata[column] = None
            elif isinstance(value, pd.Timestamp):
                metadata[column] = value.isoformat()
            else:
                metadata[column] = value

    if TARGET_COLUMN in row.index:
        target = row[TARGET_COLUMN]

        metadata[TARGET_COLUMN] = (
            None
            if pd.isna(target)
            else int(target)
        )

    return metadata


# ZAPIS JSON
def save_json(
    data: dict[str, Any],
    output_file: Path,
) -> None:
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            default=_json_serializer,
        )

    logger.info(
        "Zapisano JSON: %s",
        output_file,
    )


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

    raise TypeError(
        f"Nieobsługiwany typ JSON: {type(value)}"
    )



# ZAPIS DATAFRAME
def save_dataframe(
    df: pd.DataFrame,
    output_file: Path,
) -> None:
    if df.empty:
        raise ValueError(
            f"Nie można zapisać pustego DataFrame: {output_file}"
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_file,
        index=False,
    )

    logger.info(
        "Zapisano tabelę XAI: %s",
        output_file,
    )