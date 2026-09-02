# Model-agnostic permutation feature importance na danych OOS

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from src.models.model_config import TEST_YEARS


logger = logging.getLogger(__name__)


def calculate_permutation_importance(model,
                                     X_test: pd.DataFrame,
                                     y_test: pd.Series,
                                     n_repeats: int = 100,
                                     random_state: int = 67) -> pd.DataFrame:
    if X_test.empty:
        raise ValueError("X_test dla permutation importance jest pusty")
    if n_repeats < 1:
        raise ValueError("n_repeats musi być większe od 0")

    y_test = pd.Series(y_test).astype(int)

    baseline_pred = model.predict(X_test).astype(int)
    baseline_prob = model.predict_proba(X_test)[:, 1]

    baseline_ba = balanced_accuracy_score(y_test, baseline_pred)
    baseline_auc = (roc_auc_score(y_test, baseline_prob) if y_test.nunique() == 2
        else np.nan)

    rng = np.random.default_rng(random_state)

    # Te same permutacje wierszy dla każdej cechy ułatwiają porównanie wyników
    permutations = [rng.permutation(len(X_test)) for _ in range(n_repeats)]

    rows = []

    for feature in X_test.columns:
        ba_drops = []
        auc_drops = []

        values = X_test[feature].to_numpy()

        for indices in permutations:
            X_permuted = X_test.copy()
            X_permuted[feature] = values[indices]

            y_pred = model.predict(X_permuted).astype(int)
            y_prob = model.predict_proba(X_permuted)[:, 1]

            permuted_ba = balanced_accuracy_score(y_test, y_pred)
            ba_drops.append(baseline_ba - permuted_ba)

            if y_test.nunique() == 2:
                permuted_auc = roc_auc_score(y_test, y_prob)
                auc_drops.append(baseline_auc - permuted_auc)

        rows.append({"Feature": str(feature),
                     "Baseline_BA": baseline_ba,
                     "Mean_BA_Drop": float(np.mean(ba_drops)),
                     "Std_BA_Drop": float(np.std(ba_drops, ddof=1)) if n_repeats > 1 else 0.0,
                     "Baseline_AUC": baseline_auc,
                     "Mean_AUC_Drop": float(np.mean(auc_drops)) if auc_drops else np.nan,
                     "Std_AUC_Drop": float(np.std(auc_drops, ddof=1)) if len(auc_drops) > 1 else 0.0,
                     "N_Repeats": n_repeats})

    result = (pd.DataFrame(rows).sort_values("Mean_AUC_Drop", ascending=False)
              .reset_index(drop=True))

    result["Rank_AUC"] = np.arange(1, len(result) + 1)

    logger.info("Obliczono permutation importance | n=%d | cechy=%d | repeats=%d",
                len(X_test),
                len(X_test.columns),
                n_repeats)

    return result


def summarize_permutation_importance(df: pd.DataFrame) -> pd.DataFrame:
    result = (df.groupby(["Variant", "Feature"], as_index=False).agg(
              Mean_BA_Drop=("Mean_BA_Drop", "mean"),
              Std_BA_Drop_Across_Folds=("Mean_BA_Drop", "std"),
              Mean_AUC_Drop=("Mean_AUC_Drop", "mean"),
              Std_AUC_Drop_Across_Folds=("Mean_AUC_Drop", "std"),
              Positive_BA_Folds=("Mean_BA_Drop", lambda x: int((x > 0).sum())),
              Positive_AUC_Folds=("Mean_AUC_Drop", lambda x: int((x > 0).sum())),
              N_Folds=("Test_Year", "nunique")))

    result["Presence_Rate"] = result["N_Folds"] / len(TEST_YEARS)
    result["Positive_AUC_Rate"] = result["Positive_AUC_Folds"] / result["N_Folds"]
    result["Positive_BA_Rate"] = result["Positive_BA_Folds"] / result["N_Folds"]

    return (result.sort_values(["Variant", "Mean_AUC_Drop"], ascending=[True, False],
        ).reset_index(drop=True))


def plot_permutation_importance(importance_df: pd.DataFrame,
                                output_file: Path,
                                metric: str = "Mean_AUC_Drop",
                                top_n: int = 15) -> None:
    if metric not in importance_df.columns:
        raise ValueError(f"Brak kolumny {metric}")

    plot_df = (importance_df.nlargest(top_n, metric).sort_values(metric))

    fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * len(plot_df))))

    ax.barh(plot_df["Feature"], plot_df[metric])
    ax.axvline(0, linewidth=1)

    label = ("Spadek ROC-AUC po permutacji" if "AUC" in metric
                else "Spadek Balanced Accuracy po permutacji")

    ax.set_xlabel(label)
    ax.set_ylabel("Cecha")
    ax.set_title("Permutation Feature Importance – dane OOS")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Zapisano wykres permutation importance: %s", output_file)