#######################################################################
# Ten plik służy do wyjaśnialności regresji logistycznej
# Składa się z dwóch części:
# 1) część odpowiedzialna za pobranie regresji logistycznej
#  i odpowiednie przygotowanie jej
# 2) Część odpowiedzialna za interpretowalność regresji logistycznej
#######################################################################




############################################
############### CZĘŚĆ I ####################
############################################


import logging
from typing import Sequence

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


logger = logging.getLogger(__name__)

MODEL_NAME = "logistic_regression"


# Model
LogisticModel = LogisticRegression | Pipeline


# Pobranie Regresji logistycznej
def get_logistic_estimator(model: LogisticModel) -> LogisticRegression:
    """
    Zwraca wytrenowany estimator LogisticRegression
    Obsługuje:
    - bezpośredni LogisticRegression,
    - pipeline, w którym LogisticRegression jest ostatnim krokiem
    """

    if isinstance(model, LogisticRegression):
        estimator = model

    elif isinstance(model, Pipeline):
        estimator = model.steps[-1][1]

        if not isinstance(estimator, LogisticRegression):
            raise TypeError(
                "Ostatnim krokiem Pipeline musi być "
                "LogisticRegression."
            )

    else:
        raise TypeError(
            "Model musi być LogisticRegression lub Pipeline."
        )

    if not hasattr(estimator, "coef_"):
        raise ValueError(
            "LogisticRegression nie jest wytrenowany."
        )

    if len(estimator.classes_) != 2:
        raise ValueError(
            "XAI obsługuje tutaj wyłącznie "
            "binarną regresję logistyczną."
        )

    if not np.array_equal(
        estimator.classes_,
        np.array([0, 1]),
    ):
        raise ValueError(
            "Oczekiwano klas binarnych [0, 1], "
            f"otrzymano {estimator.classes_.tolist()}."
        )

    return estimator



# TRANSFORMACJA DANYCH DO POSTACI WIDZIANEJ PRZEZ MODEL
def transform_model_input(
    model: LogisticModel,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Zwraca X dokładnie w postaci przekazywanej
    do LogisticRegression

    Jeżeli model jest Pipeline, wykonywany jest cały preprocessing dla LogisticRegression
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "X musi być obiektem pandas.DataFrame"
        )

    if X.empty:
        raise ValueError(
            "X przekazane do XAI jest puste"
        )

    if isinstance(model, Pipeline) and len(model.steps) > 1:
        transformed = model[:-1].transform(X)
    else:
        transformed = X.to_numpy()

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(
        transformed,
        dtype=np.float64,
    )

    if transformed.ndim != 2:
        raise ValueError(
            "Dane po preprocessingu muszą mieć "
            "postać macierzy 2D."
        )

    if not np.isfinite(transformed).all():
        raise ValueError(
            "Dane po preprocessingu zawierają NaN lub Inf."
        )

    return transformed



# NAZWY CECH
def resolve_feature_names(
    X: pd.DataFrame,
    transformed_X: np.ndarray,
    feature_names: Sequence[str] | None = None,
) -> list[str]:
    """
    Ustala nazwy cech odpowiadające kolumnom
    przekazywanym do LogisticRegression.
    """

    if feature_names is None:
        names = X.columns.astype(str).tolist()
    else:
        names = [str(name) for name in feature_names]

    if len(names) != transformed_X.shape[1]:
        raise ValueError(
            "Liczba nazw cech nie odpowiada liczbie cech "
            "po preprocessingu: "
            f"names={len(names)}, "
            f"features={transformed_X.shape[1]}."
        )

    return names


# GLOBALNE WSPÓŁCZYNNIKI MODELU
def extract_coefficients(
    model: LogisticModel,
    X_reference: pd.DataFrame,
    feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Zwraca globalną interpretację współczynników regresji

    Coefficient_Log_Odds, odds_Ratio
    """

    estimator = get_logistic_estimator(model)

    transformed_X = transform_model_input(
        model=model,
        X=X_reference,
    )

    names = resolve_feature_names(
        X=X_reference,
        transformed_X=transformed_X,
        feature_names=feature_names,
    )

    coefficients = estimator.coef_[0].astype(
        np.float64
    )

    if len(coefficients) != len(names):
        raise ValueError(
            "Liczba współczynników LogisticRegression "
            "nie odpowiada liczbie cech."
        )

    result = pd.DataFrame({
        "Feature": names,
        "Coefficient_Log_Odds": coefficients,
        "Odds_Ratio": np.exp(coefficients),
        "Abs_Coefficient": np.abs(coefficients),
    })

    result["Odds_Change_Percent"] = (
        result["Odds_Ratio"] - 1.0
    ) * 100.0

    result["Direction"] = np.select(
        [
            result["Coefficient_Log_Odds"] > 0,
            result["Coefficient_Log_Odds"] < 0,
        ],
        [
            "toward_class_1",
            "toward_class_0",
        ],
        default="neutral",
    )

    result = result.sort_values(
        "Abs_Coefficient",
        ascending=False,
        ignore_index=True,
    )

    result["Rank"] = np.arange(
        1,
        len(result) + 1,
        dtype=np.int64,
    )

    

    return result


# LOKALNE WKŁADY CECH
def calculate_local_contributions(
    model: LogisticModel,
    X_row: pd.DataFrame,
    feature_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """
    Rozkłada predykcję pojedynczej obserwacji na:

        intercept + sum(beta_j * x_j) = log-odds

    Zwraca:
    - tabelę wkładów poszczególnych cech,
    - podsumowanie końcowej predykcji.
    """

    if len(X_row) != 1:
        raise ValueError(
            "Lokalne wyjaśnienie wymaga dokładnie "
            "jednej obserwacji."
        )

    estimator = get_logistic_estimator(model)

    transformed_X = transform_model_input(
        model=model,
        X=X_row,
    )

    names = resolve_feature_names(
        X=X_row,
        transformed_X=transformed_X,
        feature_names=feature_names,
    )

    model_values = transformed_X[0]
    coefficients = estimator.coef_[0].astype(
        np.float64
    )
    intercept = float(estimator.intercept_[0])

    contributions = (
        model_values * coefficients
    )

    log_odds = float(
        intercept + contributions.sum()
    )

    # Pipeline sam wykonuje preprocessing przed predict_proba.
    probability = float(
        model.predict_proba(X_row)[0, 1]
    )

    prediction = int(
        model.predict(X_row)[0]
    )

    model_log_odds = float(
        model.decision_function(X_row)[0]
    )

    if not np.isclose(
        log_odds,
        model_log_odds,
        rtol=1e-8,
        atol=1e-10,
    ):
        raise ValueError(
            "Ręcznie obliczone log-odds nie zgadzają się "
            "z decision_function modelu."
        )

    odds = float(
        np.exp(log_odds)
    )

    raw_values = X_row.iloc[0].to_numpy()

    if len(raw_values) != len(names):
        raw_values = np.full(
            len(names),
            np.nan,
        )

    contribution_df = pd.DataFrame({
        "Feature": names,
        "Raw_Feature_Value": raw_values,
        "Model_Input_Value": model_values,
        "Coefficient_Log_Odds": coefficients,
        "Log_Odds_Contribution": contributions,
        "Abs_Log_Odds_Contribution": np.abs(
            contributions
        ),
    })

    contribution_df = contribution_df.sort_values(
        "Abs_Log_Odds_Contribution",
        ascending=False,
        ignore_index=True,
    )

    contribution_df["Rank"] = np.arange(
        1,
        len(contribution_df) + 1,
        dtype=np.int64,
    )

    summary = {
        "Intercept_Log_Odds": intercept,
        "Final_Log_Odds": log_odds,
        "Final_Odds": odds,
        "Probability_Class_1": probability,
        "Prediction": prediction,
    }

    return contribution_df, summary



############################################
############### CZĘŚĆ II ###################
############################################


import matplotlib.pyplot as plt

from .common import (
    get_model_xai_dir,
    save_dataframe,
)


# WYKRES WSPÓŁCZYNNIKÓW LOG-ODDS
def plot_coefficients(
    coefficients_df: pd.DataFrame,
    output_file: Path,
    top_n: int | None = None,
) -> None:
    plot_df = coefficients_df.copy()

    if top_n is not None:
        plot_df = (
            plot_df
            .nlargest(top_n, "Abs_Coefficient")
            .sort_values("Coefficient_Log_Odds")
        )
    else:
        plot_df = plot_df.sort_values(
            "Coefficient_Log_Odds"
        )

    fig_height = max(
        5.0,
        0.45 * len(plot_df),
    )

    fig, ax = plt.subplots(
        figsize=(10, fig_height)
    )

    ax.barh(
        plot_df["Feature"],
        plot_df["Coefficient_Log_Odds"],
    )

    ax.axvline(
        0.0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Współczynnik regresji logistycznej (log-odds)"
    )
    ax.set_ylabel("Cecha")
    ax.set_title(
        "Globalny wpływ cech – regresja logistyczna"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    fig.tight_layout()

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info(
        "Zapisano wykres współczynników: %s",
        output_file,
    )


# WYKRES ODDS RATIO
def plot_odds_ratios(
    coefficients_df: pd.DataFrame,
    output_file: Path,
    top_n: int | None = None,
) -> None:
    plot_df = coefficients_df.copy()

    if top_n is not None:
        plot_df = plot_df.nlargest(
            top_n,
            "Abs_Coefficient",
        )

    plot_df = plot_df.sort_values(
        "Odds_Ratio"
    )

    fig_height = max(
        5.0,
        0.45 * len(plot_df),
    )

    fig, ax = plt.subplots(
        figsize=(10, fig_height)
    )

    ax.barh(
        plot_df["Feature"],
        plot_df["Odds_Ratio"],
    )

    ax.axvline(
        1.0,
        linewidth=1,
    )

    ax.set_xscale("log")

    ax.set_xlabel("Odds ratio")
    ax.set_ylabel("Cecha")
    ax.set_title(
        "Ilorazy szans – regresja logistyczna"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    fig.tight_layout()

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info(
        "Zapisano wykres odds ratio: %s",
        output_file,
    )


# GLOBALNE XAI REGRESJI LOGISTYCZNEJ
def run_global_logistic_xai(
    model: LogisticModel,
    X_reference: pd.DataFrame,
    feature_names: Sequence[str] | None = None,
    top_n: int = 15,
) -> pd.DataFrame:
    output_dir = get_model_xai_dir(
        MODEL_NAME
    )

    coefficients_df = extract_coefficients(
        model=model,
        X_reference=X_reference,
        feature_names=feature_names,
    )

    save_dataframe(
        coefficients_df,
        output_dir / "coefficients.csv",
    )

    plot_coefficients(
        coefficients_df=coefficients_df,
        output_file=(
            output_dir
            / "coefficients_log_odds.png"
        ),
        top_n=top_n,
    )

    plot_odds_ratios(
        coefficients_df=coefficients_df,
        output_file=(
            output_dir
            / "odds_ratios.png"
        ),
        top_n=top_n,
    )

    logger.info(
        "Zakończono globalne XAI regresji logistycznej"
    )

    return coefficients_df