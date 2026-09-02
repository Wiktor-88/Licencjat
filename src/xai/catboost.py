# SHAP dla CatBoosta

import logging

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.xai.common import (build_event_metadata, get_local_xai_dir,
    get_model_xai_dir, save_dataframe, save_json)
from src.xai.shap_utils import (build_local_shap_table, calculate_shap_importance,
    plot_shap_beeswarm, plot_shap_importance, plot_shap_waterfall,
    validate_shap_additivity)


logger = logging.getLogger(__name__)


def calculate_catboost_shap(model: CatBoostClassifier, 
                            X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pool = Pool(X, cat_features=["Ticker"])

    shap_full = np.asarray(model.get_feature_importance(pool,
                                                        type="ShapValues"),
                           dtype=float)

    shap_values = shap_full[:, :-1]
    base_values = shap_full[:, -1]

    raw_output = np.asarray(
        model.predict(pool, prediction_type="RawFormulaVal"), dtype=float).reshape(-1)

    validate_shap_additivity(shap_values, base_values, raw_output)

    return shap_values, base_values


def run_global_catboost_xai(model: CatBoostClassifier,
                            X_test: pd.DataFrame,
                            xai_name: str) -> pd.DataFrame:
    output_dir = get_model_xai_dir(xai_name)
    feature_names = X_test.columns.astype(str).tolist()

    shap_values, _ = calculate_catboost_shap(model, X_test)

    importance = calculate_shap_importance(shap_values, feature_names)

    save_dataframe(importance, output_dir / "shap_importance.csv")

    plot_shap_importance(importance, output_dir / "shap_importance.png")

    # Bez values, bo Ticker jest tekstową kategorią
    plot_shap_beeswarm(shap_values, None, feature_names, output_dir / "shap_beeswarm.png")

    return importance


def run_local_catboost_xai(model: CatBoostClassifier,
                          X_row: pd.DataFrame,
                          event_row: pd.Series,
                          xai_name: str) -> pd.DataFrame:
    feature_names = X_row.columns.astype(str).tolist()

    shap_values, base_values = calculate_catboost_shap(model, X_row)

    feature_values = X_row.iloc[0].to_numpy()

    local = build_local_shap_table(shap_values[0], feature_values, feature_names)

    accession = str(event_row["Accession"])
    output_dir = get_local_xai_dir(xai_name, accession)

    save_dataframe(local, output_dir / "shap_values.csv")

    plot_shap_waterfall(shap_values[0],
                        base_values[0],
                        feature_values,
                        feature_names,
                        output_dir / "shap_waterfall.png")

    probability = float(model.predict_proba(X_row)[0, 1])

    summary = {"Base_Value": float(base_values[0]),
               "Raw_Margin": float(model.predict(X_row, prediction_type="RawFormulaVal")[0]),
               "Probability_Class_1": probability,
               "Prediction": int(probability >= 0.5)}

    event = build_event_metadata(event_row)

    if "XAI_Example_Type" in event_row.index:
        event["XAI_Example_Type"] = str(event_row["XAI_Example_Type"])

    save_json({"event": event, "prediction": summary}, output_dir / "summary.json")

    return local