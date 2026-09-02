# XAI dla regresji logistycznej - plik przygotowujący

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.xai.common import (build_event_metadata, get_local_xai_dir, get_model_xai_dir,
    save_dataframe, save_json)


logger = logging.getLogger(__name__)

LogisticModel = LogisticRegression | Pipeline


def get_logistic_estimator(model: LogisticModel) -> LogisticRegression:
    if isinstance(model, LogisticRegression):
        estimator = model
    elif isinstance(model, Pipeline):
        estimator = model.steps[-1][1]
    else:
        raise TypeError("Model musi być LogisticRegression lub Pipeline")

    if not isinstance(estimator, LogisticRegression):
        raise TypeError("Końcowym estymatorem musi być LogisticRegression")
    if not hasattr(estimator, "coef_"):
        raise ValueError("LogisticRegression nie jest wytrenowany")
    if not np.array_equal(estimator.classes_, [0, 1]):
        raise ValueError(f'Inne klasy niż [0, 1]')

    return estimator


def transform_model_input(model: LogisticModel, X: pd.DataFrame) -> np.ndarray:
    if not isinstance(X, pd.DataFrame) or X.empty:
        raise ValueError("X musi być niepustym DataFrame")

    transformed = (model[:-1].transform(X)
                   if isinstance(model, Pipeline) and len(model.steps) > 1 else X.to_numpy())

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(transformed, dtype=float)

    if transformed.ndim != 2 or not np.isfinite(transformed).all():
        raise ValueError("Niepoprawne dane po preprocessingu")

    return transformed


def resolve_feature_names(model: LogisticModel,
                          X: pd.DataFrame,
                          transformed_X: np.ndarray,
                          feature_names: list[str] | None = None) -> list[str]:
    if feature_names is not None:
        names = [str(name) for name in feature_names]

    elif isinstance(model, Pipeline) and len(model.steps) > 1:
        try:
            names = model[:-1].get_feature_names_out().astype(str).tolist()
        
        except (AttributeError, ValueError):
            names = X.columns.astype(str).tolist()

    else:
        names = X.columns.astype(str).tolist()

    if len(names) != transformed_X.shape[1]:
        raise ValueError(f"Liczba nazw cech ({len(names)}) nie odpowiada "
                         f"liczbie cech modelu ({transformed_X.shape[1]})")

    return names


def extract_coefficients(model: LogisticModel,
                         X_reference: pd.DataFrame,
                         feature_names: list[str] | None = None) -> pd.DataFrame:
    estimator = get_logistic_estimator(model)
    transformed_X = transform_model_input(model, X_reference)

    names = resolve_feature_names(model,
                                  X_reference,
                                  transformed_X,
                                  feature_names)

    coefficients = estimator.coef_[0].astype(float)

    result = pd.DataFrame({"Feature": names,
                           "Coefficient_Log_Odds": coefficients,
                           "Abs_Coefficient": np.abs(coefficients),
                           "Odds_Ratio_Per_Model_Unit": np.exp(coefficients)})

    result["Direction"] = np.select(
        [result["Coefficient_Log_Odds"] > 0, result["Coefficient_Log_Odds"] < 0],
        ["toward_class_1", "toward_class_0"],
        default="neutral")

    return (result.sort_values("Abs_Coefficient", ascending=False)
        .reset_index(drop=True).assign(Rank=lambda x: np.arange(1, len(x) + 1)))


def calculate_local_contributions(model: LogisticModel,
                                  X_row: pd.DataFrame,
                                  feature_names: list[str] | None = None,) -> tuple[pd.DataFrame, dict]:
    if len(X_row) != 1:
        raise ValueError("Lokalne XAI wymaga dokładnie jednej obserwacji")

    estimator = get_logistic_estimator(model)
    transformed_X = transform_model_input(model, X_row)

    names = resolve_feature_names(model, X_row, transformed_X, feature_names)

    values = transformed_X[0]
    coefficients = estimator.coef_[0].astype(float)
    contributions = values * coefficients

    intercept = float(estimator.intercept_[0])
    log_odds = float(intercept + contributions.sum())
    model_log_odds = float(model.decision_function(X_row)[0])

    if not np.isclose(log_odds, model_log_odds, rtol=1e-8, atol=1e-10):
        raise ValueError("Ręczne log-odds nie zgadzają się z modelem")

    probability = float(model.predict_proba(X_row)[0, 1])
    prediction = int(model.predict(X_row)[0])

    result = pd.DataFrame({"Feature": names,
                           "Model_Input_Value": values,
                           "Coefficient_Log_Odds": coefficients,
                           "Log_Odds_Contribution": contributions,
                           "Abs_Log_Odds_Contribution": np.abs(contributions)})

    result["Direction"] = np.select(
        [result["Log_Odds_Contribution"] > 0, result["Log_Odds_Contribution"] < 0],
        ["toward_class_1", "toward_class_0"],
        default="neutral")

    result = (result.sort_values("Abs_Log_Odds_Contribution", ascending=False)
        .reset_index(drop=True).assign(Rank=lambda x: np.arange(1, len(x) + 1)))

    summary = {"Intercept_Log_Odds": intercept,
               "Final_Log_Odds": log_odds,
               "Probability_Class_1": probability,
               "Prediction": prediction}

    return result, summary


def plot_coefficients(coefficients_df: pd.DataFrame,
                      output_file: Path,
                      top_n: int = 15) -> None:
    plot_df = coefficients_df.nlargest(top_n, "Abs_Coefficient").sort_values("Coefficient_Log_Odds")
    

    fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * len(plot_df))))

    ax.barh(plot_df["Feature"], plot_df["Coefficient_Log_Odds"])
    ax.axvline(0, linewidth=1)

    ax.set_xlabel("Współczynnik log-odds")
    ax.set_ylabel("Cecha")
    ax.set_title("Regresja logistyczna – współczynniki modelu")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Zapisano wykres: %s", output_file)


def run_global_logistic_xai(model: LogisticModel,
                            X_reference: pd.DataFrame,
                            xai_name: str,
                            feature_names: list[str] | None = None,
                            top_n: int = 15) -> pd.DataFrame:
    output_dir = get_model_xai_dir(xai_name)

    coefficients = extract_coefficients(model, X_reference, feature_names)

    save_dataframe(coefficients, output_dir / "coefficients.csv")

    plot_coefficients(coefficients, output_dir / "coefficients.png", top_n=top_n)

    logger.info("Zakończono globalne XAI: %s", xai_name)

    return coefficients


def run_local_logistic_xai(model: LogisticModel,
                           X_row: pd.DataFrame,
                           event_row: pd.Series,
                           xai_name: str,
                           feature_names: list[str] | None = None) -> pd.DataFrame:
    contributions, summary = calculate_local_contributions(model, X_row, feature_names)

    accession = str(event_row["Accession"])
    output_dir = get_local_xai_dir(xai_name, accession)

    save_dataframe( contributions, output_dir / "contributions.csv",)

    plot_local_contributions(contributions, output_dir / "contributions.png")

    event = build_event_metadata(event_row)

    if "XAI_Example_Type" in event_row.index:
        event["XAI_Example_Type"] = str(event_row["XAI_Example_Type"])

    save_json({"event": event,"prediction": summary,},
            output_dir / "summary.json")

    return contributions


def plot_local_contributions(contributions_df: pd.DataFrame, 
                             output_file: Path,
                             top_n: int = 15) -> None:
    plot_df = (contributions_df.nlargest(top_n, "Abs_Log_Odds_Contribution")
        .sort_values("Log_Odds_Contribution"))

    fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * len(plot_df))))

    ax.barh(plot_df["Feature"], plot_df["Log_Odds_Contribution"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Wkład do log-odds")
    ax.set_ylabel("Cecha")
    ax.set_title("Lokalne wyjaśnienie predykcji – regresja logistyczna")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Zapisano lokalny wykres: %s", output_file)