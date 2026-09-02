# SHAP dla XGBoosta

import logging

import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

from src.xai.common import (build_event_metadata, get_local_xai_dir, get_model_xai_dir,
    save_dataframe, save_json)
from src.xai.shap_utils import (build_local_shap_table, calculate_shap_importance,
    normalize_shap_values, plot_shap_beeswarm, plot_shap_importance, plot_shap_waterfall,
    validate_shap_additivity)


logger = logging.getLogger(__name__)


def calculate_xgboost_shap(model: XGBClassifier,
                           X_processed: np.ndarray,
                           feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    explainer = shap.TreeExplainer(model, model_output="raw",)

    explanation = explainer(X_processed)
    shap_values = normalize_shap_values(explanation.values, len(feature_names))

    base_values = np.asarray(explanation.base_values, dtype=float).reshape(-1)

    raw_output = np.asarray(model.predict(X_processed, output_margin=True), dtype=float).reshape(-1)

    validate_shap_additivity(shap_values, base_values, raw_output)

    return shap_values, base_values


def run_global_xgboost_xai(model: XGBClassifier,
                           X_processed: np.ndarray,
                           feature_names: list[str],
                           xai_name: str) -> pd.DataFrame:
    output_dir = get_model_xai_dir(xai_name)

    shap_values, _ = calculate_xgboost_shap(model, X_processed, feature_names)

    importance = calculate_shap_importance(shap_values, feature_names)

    save_dataframe( importance, output_dir / "shap_importance.csv")

    plot_shap_importance(importance, output_dir / "shap_importance.png",)

    plot_shap_beeswarm(shap_values, X_processed, feature_names, output_dir / "shap_beeswarm.png")

    return importance


def run_local_xgboost_xai(model: XGBClassifier,
                          preprocessor: ColumnTransformer,
                          X_row: pd.DataFrame,
                          event_row: pd.Series,
                          xai_name: str) -> pd.DataFrame:
    X_processed = np.asarray(preprocessor.transform(X_row), dtype=float,)

    feature_names = preprocessor.get_feature_names_out().astype(str).tolist()
    

    shap_values, base_values = calculate_xgboost_shap(model, 
                                                      X_processed, 
                                                      feature_names)

    local = build_local_shap_table(shap_values[0], X_processed[0], feature_names)

    accession = str(event_row["Accession"])
    output_dir = get_local_xai_dir(xai_name, accession)

    save_dataframe(local, output_dir / "shap_values.csv",)

    plot_shap_waterfall(shap_values[0], 
                        base_values[0], 
                        X_processed[0],
                        feature_names,
                        output_dir / "shap_waterfall.png")

    raw_margin = float(model.predict(X_processed, output_margin=True)[0])

    summary = {"Base_Value": float(base_values[0]),
               "Raw_Margin": raw_margin,
              "Probability_Class_1": float(model.predict_proba(X_processed)[0, 1]),
              "Prediction": int(model.predict(X_processed)[0])}

    event = build_event_metadata(event_row)

    if "XAI_Example_Type" in event_row.index:
        event["XAI_Example_Type"] = str(event_row["XAI_Example_Type"])

    save_json({"event": event, "prediction": summary}, output_dir / "summary.json")

    return local