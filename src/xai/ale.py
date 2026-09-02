# Plik do liczenia ALE - Accumulated Local Effects dla cech numerycznych

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


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
    valid = values.notna()

    X_valid = X.loc[valid].copy()
    values = values.loc[valid].to_numpy(dtype=float)

    if len(values) < n_bins:
        raise ValueError(f"Za mało obserwacji dla ALE cechy {feature}")

    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))

    if len(edges) < 3:
        raise ValueError(f"Cecha {feature} ma za mało różnych wartości")

    bin_ids = np.searchsorted(edges[1:-1], values, side="right")

    rows = []

    for bin_id in range(len(edges) - 1):
        mask = bin_ids == bin_id
        count = int(mask.sum())

        if count == 0:
            continue

        lower = float(edges[bin_id])
        upper = float(edges[bin_id + 1])

        X_bin = X_valid.iloc[np.where(mask)[0]].copy()

        X_lower = X_bin.copy()
        X_upper = X_bin.copy()

        X_lower[feature] = lower
        X_upper[feature] = upper

        p_lower = model.predict_proba(X_lower)[:, 1]
        p_upper = model.predict_proba(X_upper)[:, 1]

        local_effect = float(np.mean(p_upper - p_lower))

        rows.append({"Feature": feature,
                     "Bin": bin_id,
                     "Bin_Lower": lower,
                     "Bin_Upper": upper,
                     "Bin_Center": (lower + upper) / 2,
                     "Local_Effect": local_effect,
                     "Count": count})

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError(f"Nie udało się obliczyć ALE dla {feature}")

    # Wartość w środku przedziału
    cumulative = result["Local_Effect"].cumsum()
    result["ALE"] = cumulative - 0.5 * result["Local_Effect"]

    # ALE centrujemy względem zera
    center = np.average(result["ALE"], weights=result["Count"])

    result["ALE"] -= center

    logger.info("Obliczono ALE | %s | n=%d | bins=%d",
                feature,
                len(values),
                len(result))

    return result


def plot_ale_by_fold(ale_df: pd.DataFrame,
                     feature: str,
                     output_file: Path) -> None:
    plot_df = ale_df[ale_df["Feature"] == feature].copy()

    if plot_df.empty:
        raise ValueError(f"Brak wyników ALE dla {feature}")

    fig, ax = plt.subplots(figsize=(9, 6))

    for test_year, year_df in plot_df.groupby("Test_Year"):
        year_df = year_df.sort_values("Bin_Center")

        ax.plot(year_df["Bin_Center"], year_df["ALE"],
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