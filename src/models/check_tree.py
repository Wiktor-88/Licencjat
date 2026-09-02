# Plik dodatkowy - test parametrów drzewa w okresie 2020-2022

import logging
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from src.models.model_config import (CATEGORICAL_FEATURES, TABULAR_MARKET_FEATURES,
    MIN_SEC_COUNT, RANDOM_STATE, SEC_BINARY_CANDIDATES, SENTIMENT_FEATURES,
    SENTIMENT_HISTORY_FLAG, TARGET)
from src.models.model_utils import calculate_metrics, prepare_model_dataset, select_sec_features



logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "model_dataset.csv"

DEPTH_VALUES = [2, 3, 4, 5, 6]
LEAF_VALUES = [5, 8, 12, 20]



def evaluate_params(df, max_depth, min_leaf):
    scores = []

    # Wewnętrzny walk-forward tylko w okresie 2020-2022
    for year in [2021, 2022]:
        train_df = df[df["Event_Session"].dt.year < year].copy()
        val_df = df[df["Event_Session"].dt.year == year].copy()

        selected_sec = select_sec_features(train_df,
                                           candidates=SEC_BINARY_CANDIDATES,
                                           min_count=MIN_SEC_COUNT)
        
        binary_features = selected_sec + [SENTIMENT_HISTORY_FLAG]
        features = (TABULAR_MARKET_FEATURES + CATEGORICAL_FEATURES
            + binary_features + SENTIMENT_FEATURES)

        preprocessor = ColumnTransformer([
            ("market", "passthrough", TABULAR_MARKET_FEATURES),
            ("ticker", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("binary", "passthrough", binary_features),
            ("sentiment", SimpleImputer(strategy="constant", fill_value=0.0), SENTIMENT_FEATURES),
        ])

        model = Pipeline([("preprocessor", preprocessor),
                        ("classifier", DecisionTreeClassifier(max_depth=max_depth,
                                                              min_samples_leaf=min_leaf,
                                                              random_state=67,
                                                              class_weight=None))])

        model.fit(train_df[features], train_df[TARGET])

        y_pred = model.predict(val_df[features])
        y_prob = model.predict_proba(val_df[features])[:, 1]

        metrics = calculate_metrics(val_df[TARGET], y_pred, y_prob)

        scores.append({"BA": metrics["Balanced_Accuracy"],
                       "ROC_AUC": metrics["ROC_AUC"]})

    mean_ba = sum(score["BA"] for score in scores) / len(scores)
    mean_auc = sum(score["ROC_AUC"] for score in scores) / len(scores)

    return mean_ba, mean_auc


def main():
    df = pd.read_csv(DATA_FILE)
    df = prepare_model_dataset(df, TARGET)
    df = df[df["Event_Session"].dt.year <= 2022].copy()

    results = []

    for depth in DEPTH_VALUES:
        for leaf in LEAF_VALUES:
            ba, auc = evaluate_params(df, depth, leaf)

            results.append({"max_depth": depth,
                            "min_samples_leaf": leaf,
                            "Balanced_Accuracy": ba,
                            "ROC_AUC": auc})

    results_df = pd.DataFrame(results).sort_values(["Balanced_Accuracy", "ROC_AUC"], ascending=False)

    logger.info("Wyniki strojenia:\n%s", results_df.to_string(index=False))

    best = results_df.iloc[0]

    logger.info("Najlepsze parametry: max_depth=%d | min_samples_leaf=%d | BA=%.4f | ROC-AUC=%.4f",
                best["max_depth"],
                best["min_samples_leaf"],
                best["Balanced_Accuracy"],
                best["ROC_AUC"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    main()
