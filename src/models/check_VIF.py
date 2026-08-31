# Plik dodatkowy - sprawdzjący VIF, w celu wybrania cech dla regresji logistycznej 

import pandas as pd
import logging
import numpy as np

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from pathlib import Path

from src.models.model_config import MARKET_FEATURES, MARKET_Z_FEATURES, TARGET, TEST_YEARS
from src.models.model_utils import prepare_model_dataset, validate_target



logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "model_dataset.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "vif"
RAW_OUTPUT = OUTPUT_DIR / "vif_market_raw.csv"
Z60_OUTPUT = OUTPUT_DIR / "vif_market_z60.csv"

# Wszystkie dane przed pierwszym rokiem testowym
DEVELOPMENT_END_YEAR = min(TEST_YEARS) - 1


def calculate_vif(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:

    X = df[features].astype(float).copy()

    # Dodajemy wyraz wolny
    X = add_constant(X, has_constant="add",)

    rows = []

    for i, feature in enumerate(X.columns):
        if feature == "const":
            continue

        vif = variance_inflation_factor(X.to_numpy(), i)

        rows.append({"Feature": feature,
                    "VIF": float(vif)})

    result = pd.DataFrame(rows)

    result["VIF_Level"] = pd.cut(result["VIF"],
                                 bins=[-np.inf, 5, 10, np.inf],
                                 labels=["Niski", "Średni", "Wysoki"],
                                 right=False)

    return result.sort_values("VIF", ascending=False).reset_index(drop=True)


def main() -> None:
    
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    df["Event_Session"] = pd.to_datetime(df["Event_Session"])

    # VIF liczymy tylko na danych dostępnych przed pierwszym testem 
    df = df[(df["Use_In_Primary_Model"] == 1)
            & df["Target_Abnormal_1D"].notna()
            & (df["Event_Session"].dt.year <= DEVELOPMENT_END_YEAR)].copy()

    logger.info("Okres development: %s -> %s",
                 df["Event_Session"].min().date(),
                 df["Event_Session"].max().date())
    logger.info("Liczba obserwacji: %d", len(df))

    raw_vif = calculate_vif(df, MARKET_FEATURES)
    z60_vif = calculate_vif(df, MARKET_Z_FEATURES)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_vif.to_csv(RAW_OUTPUT, index=False)
    z60_vif.to_csv(Z60_OUTPUT, index=False)

    logger.info("VIF - MARKET RAW:\n%s", raw_vif.to_string(index=False))
    logger.info("VIF - MARKET Z60:\n%s", z60_vif.to_string(index=False))
    logger.info("Zapisano wyniki RAW: %s", RAW_OUTPUT)
    logger.info("Zapisano wyniki Z60: %s", Z60_OUTPUT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    main()