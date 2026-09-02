# Plik dodatkowy - testowanie parami

import logging
from pathlib import Path

import pandas as pd

from src.models.model_config import RANDOM_STATE
from src.models.stat_test_utils import (holm_adjust, prepare_prediction_pair,
                                        run_paired_permutation_test)


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = DATA_DIR / "finbert_increment_test.csv"

N_PERMUTATIONS = 5000

EXPERIMENTS = [("Logistic Regression", "oos_predictions.csv",
                "MODEL B - MARKET + SEC", "MODEL C - MARKET + SEC + FINBERT"),

                ("Decision Tree", "decision_tree_oos_predictions.csv",
                "TREE B - MARKET + SEC", "TREE C - MARKET + SEC + FINBERT"),

                ("XGBoost", "xgboost_oos_predictions.csv",
                "XGB B - MARKET + SEC", "XGB C - MARKET + SEC + FINBERT"),

                ("CatBoost", "catboost_oos_predictions.csv",
                "CAT B - MARKET + SEC", "CAT C - MARKET + SEC + FINBERT"),

                ("TabNet", "tabnet_oos_predictions.csv",
                "TABNET B - MARKET + SEC", "TABNET C - MARKET + SEC + FINBERT"),

                ("LSTM", "lstm_oos_predictions.csv",
                "LSTM B - MARKET SEQUENCE + SEC",
                "LSTM C - MARKET SEQUENCE + SEC + FINBERT"),

                ("Transformer", "transformer_oos_predictions.csv",
                "TRANSFORMER B - MARKET SEQUENCE + SEC",
                "TRANSFORMER C - MARKET SEQUENCE + SEC + FINBERT")]


def main() -> None:
    results = []

    for algorithm, filename, model_a, model_b in EXPERIMENTS:
        df = pd.read_csv(DATA_DIR / filename)
        pair_df = prepare_prediction_pair(df, model_a, model_b)

        result = run_paired_permutation_test(pair_df,
                                             n_permutations=N_PERMUTATIONS,
                                             random_state=RANDOM_STATE) 

        result.update({"Algorithm": algorithm,
                       "Model_A": model_a,
                       "Model_B": model_b})
        results.append(result)

        logger.info("%s | różne klasy: %d/%d | BA %.4f -> %.4f p=%.4f | "
                    "AUC %.4f -> %.4f p=%.4f",
                    algorithm,
                    result["N_Pred_Diff"],
                    result["N_OOS"],
                    result["Pooled_BA_A"],
                    result["Pooled_BA_B"],
                    result["P_BA"],
                    result["Pooled_AUC_A"],
                    result["Pooled_AUC_B"],
                    result["P_AUC"])

    results_df = pd.DataFrame(results)
    results_df["P_BA_Holm"] = holm_adjust(results_df["P_BA"].tolist())
    results_df["P_AUC_Holm"] = holm_adjust(results_df["P_AUC"].tolist())

    results_df.to_csv(OUTPUT_FILE, index=False)

    logger.info("Wyniki testu FinBERT:\n%s", results_df.to_string(index=False))
    logger.info("Zapisano: %s", OUTPUT_FILE)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    main()
