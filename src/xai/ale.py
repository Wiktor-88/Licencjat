# Plik do liczenia ALE - Accumulated Local Effects dla cech numerycznych

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyALE import ale


logger = logging.getLogger(__name__)


class PositiveClassProbability:
    """Adapter zwracający PyALE prawdopodobieństwo klasy dodatniej."""

    def __init__(self, model) -> None:
        self.model = model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = np.asarray(self.model.predict_proba(X))

        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ValueError("ALE wymaga klasyfikatora binarnego z predict_proba")

        return probabilities[:, 1]


def select_ale_features(pfi_df: pd.DataFrame, 
                        variant: str,
                        candidate_features: list[str],
                        top_n: int = 3) -> list[str]:
    selected = (pfi_df[(pfi_df["Variant"] == variant)
                         & pfi_df["Feature"].isin(candidate_features)]
        .sort_values("Mean_AUC_Drop", ascending=False).head(top_n)["Feature"]
        .tolist())

    if not selected:
        raise ValueError(f"Nie znaleziono cech do ALE dla wariantu {variant}")

    return selected


def calculate_ale_1d(model,
                     X: pd.DataFrame,
                     feature: str,
                     n_bins: int = 10,) -> pd.DataFrame:
    if feature not in X.columns:
        raise ValueError(f"Brak cechy {feature}")
    if n_bins < 2:
        raise ValueError("n_bins musi być >= 2")

    values = pd.to_numeric(X[feature], errors="coerce")
    X_valid = X.loc[values.notna()].copy()
    X_valid[feature] = values.loc[values.notna()].astype(float)

    if len(X_valid) < n_bins:
        raise ValueError(f"Za mało obserwacji dla ALE cechy {feature}")

    if X_valid[feature].nunique() < 2:
        raise ValueError(f"Cecha {feature} ma za mało różnych wartości")

    result = ale(X=X_valid,
                 model=PositiveClassProbability(model),
                 feature=[feature],
                 feature_type="continuous",
                 grid_size=n_bins,
                 include_CI=False,
                 plot=False)

    result = (result.reset_index()
        .rename(columns={feature: "Feature_Value", "eff": "ALE", "size": "Count"}))
    result.insert(0, "Feature", feature)
    result["Count"] = result["Count"].fillna(0).astype(int)

    if result.empty:
        raise ValueError(f"Nie udało się obliczyć ALE dla {feature}")

    logger.info("Obliczono ALE przez PyALE | %s | n=%d | grid=%d",
                feature,
                len(X_valid),
                len(result) - 1)

    return result


def plot_ale_by_fold(ale_df: pd.DataFrame,
                     feature: str,
                     output_file: Path) -> None:
    plot_df = ale_df[ale_df["Feature"] == feature].copy()

    if plot_df.empty:
        raise ValueError(f"Brak wyników ALE dla {feature}")

    fig, ax = plt.subplots(figsize=(9, 6))

    for test_year, year_df in plot_df.groupby("Test_Year"):
        year_df = year_df.sort_values("Feature_Value")

        ax.plot(year_df["Feature_Value"], year_df["ALE"],
                marker="o",
                label=str(test_year))

    ax.axhline(0, linewidth=1)
    ax.set_xlabel(feature)
    ax.set_ylabel("ALE dla P(y=1)")
    ax.set_title(f"Accumulated Local Effects – {feature}")
    ax.grid(alpha=0.25)
    ax.legend(title="Test year")

    fig.tight_layout()

    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.close(fig)

    logger.info("Zapisano wykres ALE: %s", output_file)
