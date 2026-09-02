# Wspólne funkcje SHAP dla modeli black-box

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.models.model_config import TEST_YEARS


logger = logging.getLogger(__name__)


def normalize_shap_values(shap_values, n_features: int, class_index: int = 1) -> np.ndarray:
    
    if isinstance(shap_values, list):
        shap_values = shap_values[class_index] if len(shap_values) > 1 else shap_values[0]

    values = np.asarray(shap_values, dtype=float)

    if values.ndim == 1:
        values = values.reshape(1, -1)

    elif values.ndim == 3:
        if values.shape[1] == n_features and values.shape[2] > class_index:
            values = values[:, :, class_index]
        elif values.shape[0] > class_index and values.shape[2] == n_features:
            values = values[class_index]
        else:
            raise ValueError(f"Nieobsługiwany kształt SHAP: {values.shape}")

    if values.ndim != 2 or values.shape[1] != n_features:
        raise ValueError(f"Niepoprawny kształt SHAP {values.shape}, oczekiwano (*, {n_features})")

    if not np.isfinite(values).all():
        raise ValueError("Wartości SHAP zawierają NaN lub Inf")

    return values


def validate_shap_additivity(shap_values: np.ndarray,
                             base_values,
                             raw_output,
                             atol: float = 1e-5) -> None:
    n_rows = len(shap_values)

    base = np.asarray(base_values, dtype=float)

    if base.ndim == 0:
        base = np.full(n_rows, float(base))
    else:
        base = base.reshape(-1)

    if len(base) == 1:
        base = np.full(n_rows, base[0])
    elif len(base) != n_rows:
        raise ValueError("Liczba base values nie odpowiada liczbie obserwacji")

    raw_output = np.asarray(raw_output, dtype=float).reshape(-1)

    if len(raw_output) != n_rows:
        raise ValueError("Liczba raw output nie odpowiada liczbie obserwacji")

    reconstructed = base + shap_values.sum(axis=1)

    if not np.allclose(reconstructed,
                       raw_output,
                       rtol=1e-5,
                       atol=atol):
        max_diff = float(np.max(np.abs(reconstructed - raw_output)))

        raise ValueError(f"SHAP nie odtwarza raw output modelu, max różnica={max_diff:.6g}")


def calculate_shap_importance(shap_values: np.ndarray, 
                              feature_names: list[str]) -> pd.DataFrame:
    if shap_values.shape[1] != len(feature_names):
        raise ValueError("Liczba cech nie odpowiada liczbie kolumn SHAP")

    result = pd.DataFrame({"Feature": feature_names,
                           "Mean_Abs_SHAP": np.mean(np.abs(shap_values), axis=0),
                           "Mean_SHAP": np.mean(shap_values, axis=0),
                           "Std_SHAP": np.std(shap_values, axis=0),
                           "Positive_Share": np.mean(shap_values > 0, axis=0),
                           "Negative_Share": np.mean(shap_values < 0, axis=0)})

    return (result.sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
        .assign(Rank=lambda x: np.arange(1, len(x) + 1)))


def summarize_shap_importance(df: pd.DataFrame,) -> pd.DataFrame:
    result = (df.groupby(["Variant", "Feature"], as_index=False).agg(
            Mean_Abs_SHAP=("Mean_Abs_SHAP", "mean"),
            Std_Abs_SHAP_Across_Folds=("Mean_Abs_SHAP", "std"),
            Mean_SHAP=("Mean_SHAP", "mean"),
            Mean_Positive_Share=("Positive_Share", "mean"),
            Mean_Negative_Share=("Negative_Share", "mean"),
            N_Folds=("Test_Year", "nunique")))

    result["Presence_Rate"] = result["N_Folds"] / len(TEST_YEARS)

    return (result.sort_values(["Variant", "Mean_Abs_SHAP"], ascending=[True, False],
        ).reset_index(drop=True))


def build_local_shap_table(shap_values: np.ndarray, 
                           feature_values,
                           feature_names: list[str]) -> pd.DataFrame:
    values = np.asarray(feature_values).reshape(-1)
    shap_values = np.asarray(shap_values, dtype=float).reshape(-1)

    if len(values) != len(feature_names) or len(shap_values) != len(feature_names):
        raise ValueError("Niezgodna liczba cech w local SHAP")

    result = pd.DataFrame({"Feature": feature_names,
                           "Feature_Value": values,
                           "SHAP_Value": shap_values,
                           "Abs_SHAP_Value": np.abs(shap_values)})

    result["Direction"] = np.select(
        [result["SHAP_Value"] > 0, result["SHAP_Value"] < 0],
        ["toward_class_1", "toward_class_0"],
        default="neutral")

    return (result.sort_values("Abs_SHAP_Value", ascending=False).reset_index(drop=True)
        .assign(Rank=lambda x: np.arange(1, len(x) + 1)))


# Globalny wykres SHAP
def plot_shap_importance(importance_df: pd.DataFrame,
                         output_file: Path,
                         top_n: int = 15) -> None:
    plot_df = (importance_df.nlargest(top_n, "Mean_Abs_SHAP")
        .sort_values("Mean_Abs_SHAP"))

    fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * len(plot_df))))

    ax.barh(plot_df["Feature"], plot_df["Mean_Abs_SHAP"])

    ax.set_xlabel("Średnia bezwzględna wartość SHAP")
    ax.set_ylabel("Cecha")
    ax.set_title("Globalna ważność cech według SHAP")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Zapisano SHAP importance: %s", output_file)


# Wykres BeeSwarm
def plot_shap_beeswarm(shap_values: np.ndarray,
                       feature_values,
                       feature_names: list[str],
                       output_file: Path,
                       max_display: int = 15) -> None:
    values = None

    if feature_values is not None:
        values = (feature_values.to_numpy() if isinstance(feature_values, pd.DataFrame)
                  else np.asarray(feature_values))

        if values.shape != shap_values.shape:
            raise ValueError(f"Feature values {values.shape} i SHAP {shap_values.shape} mają różne wymiary")

    explanation = shap.Explanation(values=shap_values,
                                   data=values,
                                   feature_names=feature_names)

    shap.plots.beeswarm(explanation, max_display=max_display, show=False)

    fig = plt.gcf()
    fig.tight_layout()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Zapisano SHAP beeswarm: %s", output_file)



# lokalny waterfall
def plot_shap_waterfall(shap_values,
                        base_value: float,
                        feature_values,
                        feature_names: list[str],
                        output_file: Path,
                        max_display: int = 15) -> None:
    explanation = shap.Explanation(values=np.asarray(shap_values, dtype=float).reshape(-1),
                                   base_values=float(base_value),
                                   data=np.asarray(feature_values).reshape(-1),
                                   feature_names=feature_names)

    shap.plots.waterfall(explanation,
                         max_display=max_display,
                         show=False)

    fig = plt.gcf()
    fig.tight_layout()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Zapisano SHAP waterfall: %s", output_file)