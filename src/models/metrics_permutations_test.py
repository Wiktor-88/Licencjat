"""Blokowy test permutacyjny zapisanych predykcji OOS względem braku informacji."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from src.models.model_config import RANDOM_STATE
from src.models.stat_test_utils import holm_adjust


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = DATA_DIR / "permutation_test_oos.csv"
N_PERMUTATIONS = 5000

EXPERIMENTS = {
    "Logistic Regression": ("oos_predictions.csv", ["MODEL A - MARKET", "MODEL B - MARKET + SEC", "MODEL C - MARKET + SEC + FINBERT"]),
    "Decision Tree": ("decision_tree_oos_predictions.csv", ["TREE A - MARKET", "TREE B - MARKET + SEC", "TREE C - MARKET + SEC + FINBERT"]),
    "XGBoost": ("xgboost_oos_predictions.csv", ["XGB A - MARKET", "XGB B - MARKET + SEC", "XGB C - MARKET + SEC + FINBERT"]),
    "CatBoost": ("catboost_oos_predictions.csv", ["CAT A - MARKET", "CAT B - MARKET + SEC", "CAT C - MARKET + SEC + FINBERT"]),
    "TabNet": ("tabnet_oos_predictions.csv", ["TABNET A - MARKET", "TABNET B - MARKET + SEC", "TABNET C - MARKET + SEC + FINBERT"]),
    "LSTM": ("lstm_oos_predictions.csv", ["LSTM A - MARKET SEQUENCE", "LSTM B - MARKET SEQUENCE + SEC", "LSTM C - MARKET SEQUENCE + SEC + FINBERT"]),
    "Transformer": ("transformer_oos_predictions.csv", ["TRANSFORMER A - MARKET SEQUENCE", "TRANSFORMER B - MARKET SEQUENCE + SEC", "TRANSFORMER C - MARKET SEQUENCE + SEC + FINBERT"])}


def calculate_metrics(df: pd.DataFrame, target_col: str = "y_true") -> tuple[float, float]:
    y_true = df[target_col].astype(int).to_numpy()
    return calculate_array_metrics(y_true, df["y_pred"].astype(int).to_numpy(), df["y_prob"].astype(float).to_numpy())


def calculate_array_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    if np.unique(y_true).size != 2:
        raise ValueError("ROC-AUC wymaga obu klas w zbiorze OOS.")

    balanced = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    return float(balanced), float(auc)


def calculate_fixed_prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray, probability_ranks: np.ndarray) -> tuple[float, float]:
    positive_count = int(y_true.sum())
    negative_count = len(y_true) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Metryki wymagają obu klas w zbiorze OOS.")

    true_positive_rate = y_pred[y_true == 1].mean()
    true_negative_rate = (1 - y_pred[y_true == 0]).mean()
    balanced = 0.5 * (true_positive_rate + true_negative_rate)
    auc = (probability_ranks[y_true == 1].sum() - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count)
    return float(balanced), float(auc)


def aggregate_event_decisions(df: pd.DataFrame) -> pd.DataFrame:
    """Kilka filingów tej samej spółki na tę samą sesję tworzy jedną decyzję."""
    consistency = df.groupby(["Test_Year", "Ticker", "Event_Session"])["y_true"].nunique()
    if consistency.gt(1).any():
        raise ValueError("Ten sam ticker i sesja mają różne wartości targetu.")

    events = df.groupby(["Test_Year", "Ticker", "Event_Session"], as_index=False).agg(
        y_true=("y_true", "first"), y_prob=("y_prob", "mean"), Filing_Count=("Accession", "size"))
    events["y_pred"] = events["y_prob"].ge(0.5).astype(int)
    return events.sort_values(["Test_Year", "Event_Session", "Ticker"]).reset_index(drop=True)


def build_permutation_groups(df: pd.DataFrame) -> list[np.ndarray]:
    groups = []
    for _, year_df in df.groupby("Test_Year", sort=True):
        size_groups: dict[int, list[np.ndarray]] = {}
        for _, rows in year_df.groupby("Event_Session", sort=True):
            size_groups.setdefault(len(rows), []).append(rows.index.to_numpy())
        groups.extend(np.vstack(indexes) for indexes in size_groups.values() if len(indexes) >= 2)
    return groups


def permute_targets_by_session(df: pd.DataFrame, rng: np.random.Generator,
    groups: list[np.ndarray] | None = None) -> tuple[np.ndarray, int]:
    """Przenosi całe wektory targetu między sesjami tej samej wielkości i roku."""
    permuted = df["y_true"].to_numpy(copy=True)
    groups = groups or build_permutation_groups(df)
    movable_sessions = sum(len(index_matrix) for index_matrix in groups)

    for index_matrix in groups:
        donor_order = rng.permutation(len(index_matrix))
        original = permuted[index_matrix].copy()
        permuted[index_matrix.ravel()] = original[donor_order].ravel()

    return permuted, movable_sessions


def run_permutation_test(model_df: pd.DataFrame, rng: np.random.Generator) -> dict:
    y_true = model_df["y_true"].astype(int).to_numpy()
    y_pred = model_df["y_pred"].astype(int).to_numpy()
    y_prob = model_df["y_prob"].astype(float).to_numpy()
    probability_ranks = rankdata(y_prob, method="average")
    observed_balanced, observed_auc = calculate_fixed_prediction_metrics(y_true, y_pred, probability_ranks)
    permutation_balanced = np.empty(N_PERMUTATIONS)
    permutation_auc = np.empty(N_PERMUTATIONS)
    groups = build_permutation_groups(model_df)
    movable_sessions = sum(len(index_matrix) for index_matrix in groups)

    if not groups:
        raise ValueError("Brak co najmniej dwóch sesji o tej samej liczbie zdarzeń w jednym roku.")

    for permutation_id in range(N_PERMUTATIONS):
        y_permuted, _ = permute_targets_by_session(model_df, rng, groups)
        permutation_balanced[permutation_id], permutation_auc[permutation_id] = calculate_fixed_prediction_metrics(
            y_permuted, y_pred, probability_ranks)

    p_balanced = (1 + np.sum(permutation_balanced >= observed_balanced)) / (N_PERMUTATIONS + 1)
    p_auc = (1 + np.sum(permutation_auc >= observed_auc)) / (N_PERMUTATIONS + 1)

    return {"N_Events": len(model_df), "N_Sessions": model_df["Event_Session"].nunique(), "N_Movable_Sessions": movable_sessions,
        "Observed_Balanced_Accuracy": observed_balanced, "Null_Balanced_Mean": float(permutation_balanced.mean()),
        "Null_Balanced_Std": float(permutation_balanced.std(ddof=1)), "Null_Balanced_95": float(np.quantile(permutation_balanced, 0.95)),
        "P_Value_Balanced": float(p_balanced), "Observed_ROC_AUC": observed_auc,
        "Null_ROC_Mean": float(permutation_auc.mean()), "Null_ROC_Std": float(permutation_auc.std(ddof=1)),
        "Null_ROC_95": float(np.quantile(permutation_auc, 0.95)), "P_Value_ROC": float(p_auc)}


def load_predictions() -> pd.DataFrame:
    required = {"Model", "Test_Year", "Ticker", "Event_Session", "Accession", "y_true", "y_pred", "y_prob"}
    frames = []

    for family, (filename, models) in EXPERIMENTS.items():
        path = DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku dla {family}: {path}")

        frame = pd.read_csv(path)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{family}: brak kolumn {sorted(missing)}")

        frame = frame.loc[frame["Model"].isin(models)].copy()
        found = sorted(frame["Model"].unique())
        if set(found) != set(models):
            raise ValueError(f"{family}: oczekiwano modeli {models}, znaleziono {found}")

        frame["Family"] = family
        frames.append(frame)
        logger.info("%s | modele: %s | predykcje: %d", family, found, len(frame))

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    predictions = load_predictions()
    results = []

    for (family, model_name), model_df in predictions.groupby(["Family", "Model"], sort=False):
        model_df = aggregate_event_decisions(model_df)
        result = run_permutation_test(model_df, np.random.default_rng(RANDOM_STATE))
        result.update({"Family": family, "Model": model_name})
        results.append(result)
        logger.info("%s | %s | BA=%.4f p=%.4f | AUC=%.4f p=%.4f", family, model_name,
            result["Observed_Balanced_Accuracy"], result["P_Value_Balanced"], result["Observed_ROC_AUC"], result["P_Value_ROC"])

    results_df = pd.DataFrame(results)
    results_df["P_Value_Balanced_Holm"] = holm_adjust(results_df["P_Value_Balanced"].tolist())
    results_df["P_Value_ROC_Holm"] = holm_adjust(results_df["P_Value_ROC"].tolist())
    results_df.to_csv(OUTPUT_FILE, index=False)
    logger.info("Wyniki testu permutacyjnego:\n%s", results_df.to_string(index=False))
    logger.info("Zapisano: %s", OUTPUT_FILE)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    main()
