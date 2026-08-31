# Drugi plik do trenowania - drzewo decyzjne

import numpy as np
import pandas as pd
import logging

import logging
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from model_utils import (add_confusion_metrics, calculate_metrics, create_summary,
    log_repeated_events, prepare_model_dataset, select_sec_features, validate_target)

from model_config import (CATEGORICAL_FEATURES, MARKET_COMPACT_FEATURES, MIN_SEC_COUNT,
    SEC_BINARY_CANDIDATES, SENTIMENT_FEATURES, SENTIMENT_HISTORY_FLAG,
    TARGET, TEST_YEARS)


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "model_dataset.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTIONS_FILE = OUTPUT_DIR / "decision_tree_oos_predictions.csv"
FOLD_METRICS_FILE = OUTPUT_DIR / "decision_tree_fold_metrics.csv"
SUMMARY_FILE = OUTPUT_DIR / "decision_tree_summary.csv"
IMPORTANCE_FILE = OUTPUT_DIR / "decision_tree_feature_importance.csv"



def build_tree_pipeline(numeric_features: list[str],
                        binary_features: list[str],
                        sentiment_features: list[str] | None = None) -> Pipeline:

    sentiment_features = sentiment_features or []

    transformers = [("numeric", "passthrough", numeric_features),
                    ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)]

    if binary_features:
        transformers.append(("binary", "passthrough", binary_features))

    if sentiment_features:
        sentiment_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0))
            ])

        transformers.append(("sentiment", sentiment_pipeline, sentiment_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    classifier = DecisionTreeClassifier(max_depth=4,
                                        min_samples_leaf=8,
                                        random_state=67,
                                        class_weight=None)

    return Pipeline(steps=[("preprocessor", preprocessor),
                            ("classifier", classifier)
                            ])


def evaluate_tree(model_name: str,
                train_df: pd.DataFrame,
                test_df: pd.DataFrame,
                numeric_features: list[str],
                binary_features: list[str],
                test_year: int,
                sentiment_features: list[str] | None = None) -> tuple[dict, pd.DataFrame, pd.DataFrame]:

    sentiment_features = sentiment_features or []

    input_features = list(dict.fromkeys(numeric_features
                                        + CATEGORICAL_FEATURES
                                        + binary_features
                                        + sentiment_features))

    X_train = train_df[input_features].copy()
    X_test = test_df[input_features].copy()

    y_train = train_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)

    if y_train.nunique() != 2:
        raise ValueError(f"TRAIN dla testu {test_year} nie zawiera obu klas targetu")

    model = build_tree_pipeline(numeric_features=numeric_features,
                                binary_features=binary_features,
                                sentiment_features=sentiment_features)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    feature_names =  model.named_steps["preprocessor"].get_feature_names_out()
    

    importance = pd.DataFrame({
        "Test_Year": test_year,
        "Model": model_name,
        "Feature": feature_names,
        "Importance": model.named_steps["classifier"].feature_importances_,
    })

    predictions = test_df[["Ticker", "Event_Session", "Accession", "Abnormal_Event_Return_1D"]].copy()

    predictions["Test_Year"] = test_year
    predictions["Model"] = model_name
    predictions["y_true"] = y_test.to_numpy()
    predictions["y_pred"] = y_pred
    predictions["y_prob"] = y_prob

    result = {
        "Test_Year": test_year,
        "Model": model_name,
        "Train_Size": len(train_df),
        "Test_Size": len(test_df),
        "Num_Input_Features": len(input_features),
        "Num_Encoded_Features": len(feature_names),
        "Selected_SEC_Features": "|".join(feature for feature in binary_features if feature in SEC_BINARY_CANDIDATES),
        **calculate_metrics(y_test, y_pred, y_prob),
    }

    add_confusion_metrics(result=result, y_true=y_test, y_pred=y_pred)

    return result, predictions, importance




def build_model_configs(selected_sec: list[str]) -> list[dict]:
    return [{"name": "TREE A - COMPACT MARKET",
            "numeric": MARKET_COMPACT_FEATURES,
            "binary": [],
            "sentiment": []},

            {"name": "TREE B - COMPACT + SEC",
            "numeric": MARKET_COMPACT_FEATURES,
            "binary": selected_sec,
            "sentiment": []},

            {"name": "TREE C - COMPACT + SEC + FINBERT",
            "numeric": MARKET_COMPACT_FEATURES,
            "binary": selected_sec + [SENTIMENT_HISTORY_FLAG],
            "sentiment": SENTIMENT_FEATURES}]



def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    df = prepare_model_dataset(df, TARGET)

    validate_target(df, TARGET)
    log_repeated_events(df)

    logger.info("Target: %s", TARGET)
    logger.info("Liczba obserwacji primary: %d", len(df))
    logger.info("Zakres danych: %s -> %s",
                df["Event_Session"].min().date(),
                df["Event_Session"].max().date())
    logger.info("Rozkład targetu:\n%s",
                df[TARGET].value_counts().sort_index().to_string())

    results = []
    all_predictions = []
    all_importances = []

    for test_year in TEST_YEARS:
        train_df = df[df["Event_Session"].dt.year < test_year].copy()

        test_df = df[df["Event_Session"].dt.year == test_year].copy()

        if train_df.empty or test_df.empty:
            logger.warning("Pominięcie foldu dla roku %d - pusty TRAIN lub TEST", test_year)
            continue

        logger.info("FOLD %d | TRAIN=%d (%s -> %s) | TEST=%d (%s -> %s)",
                    test_year,
                    len(train_df),
                    train_df["Event_Session"].min().date(),
                    train_df["Event_Session"].max().date(),
                    len(test_df),
                    test_df["Event_Session"].min().date(),
                    test_df["Event_Session"].max().date())

        logger.info("Target TRAIN:\n%s",
                    train_df[TARGET].value_counts().sort_index().to_string())

        logger.info("Target TEST:\n%s",
                    test_df[TARGET].value_counts().sort_index().to_string())

        # Flagi SEC wybieramy tylko na TRAIN
        selected_sec = select_sec_features(train_df,
                                            candidates=SEC_BINARY_CANDIDATES,
                                            min_count=MIN_SEC_COUNT)

        sec_counts = {feature: int(train_df[feature].sum()) for feature in selected_sec}

        logger.info("SEC features dla TEST %d:\n%s",
                     test_year, 
                     pd.Series(sec_counts).to_string())

        for config in build_model_configs(selected_sec):
            result, predictions, importance = evaluate_tree(model_name=config["name"],
                                                            train_df=train_df,
                                                            test_df=test_df,
                                                            numeric_features=config["numeric"],
                                                            binary_features=config["binary"],
                                                            sentiment_features=config["sentiment"],
                                                            test_year=test_year)

            results.append(result)
            all_predictions.append(predictions)
            all_importances.append(importance)

            logger.info("%s | TEST %d | ACC=%.4f | BA=%.4f | F1=%.4f | ROC-AUC=%.4f",
                        config["name"],
                        test_year,
                        result["Accuracy"],
                        result["Balanced_Accuracy"],
                        result["F1"],
                        result["ROC_AUC"])

            logger.info("%s | TEST %d | TN=%d FP=%d FN=%d TP=%d",
                        config["name"],
                        test_year,
                        result["TN"],
                        result["FP"],
                        result["FN"],
                        result["TP"])

    if not results:
        raise RuntimeError("Nie udało się wytrenować żadnego drzewa")

    results_df = pd.DataFrame(results)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    importance_df = pd.concat(all_importances, ignore_index=True)

    summary_df = create_summary(results_df=results_df, predictions_df=predictions_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(FOLD_METRICS_FILE, index=False)
    predictions_df.to_csv(PREDICTIONS_FILE, index=False)
    importance_df.to_csv(IMPORTANCE_FILE, index=False)
    summary_df.to_csv(SUMMARY_FILE, index=False)

    logger.info("Średnie wyniki walk-forward:\n%s",
                summary_df.to_string(index=False))

    logger.info("Metryki foldów: %s", FOLD_METRICS_FILE)
    logger.info("Predykcje OOS: %s", PREDICTIONS_FILE)
    logger.info("Feature importance: %s", IMPORTANCE_FILE)
    logger.info("Podsumowanie: %s", SUMMARY_FILE)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    main()