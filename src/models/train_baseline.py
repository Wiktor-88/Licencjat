# Pierwszy lik do trenowania modelu bazowego, czyli regreji logistycznej w kilu różnych wariantach

import numpy as np
import pandas as pd
import logging

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from pathlib import Path

from src.models.model_utils import (add_confusion_metrics, calculate_metrics,
    create_summary, log_repeated_events, prepare_model_dataset, select_sec_features,
    validate_target, add_sentiment_context)

from src.models.model_config import (
    CATEGORICAL_FEATURES, MARKET_COMPACT_FEATURES, MARKET_FEATURES, MARKET_Z_FEATURES,
    MIN_SEC_COUNT, SEC_BINARY_CANDIDATES, SENTIMENT_CONTEXT_FEATURES, SENTIMENT_FEATURES,
    SENTIMENT_HISTORY_FLAG, TARGET, TEST_YEARS)


# Zmiana nazwy
MIN_SEC_OCCURRENCES = MIN_SEC_COUNT

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "model_dataset.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTIONS_FILE = OUTPUT_DIR / "oos_predictions.csv"
FOLD_METRICS_FILE = OUTPUT_DIR / "logistic_fold_metrics.csv"
SUMMARY_FILE = OUTPUT_DIR / "logistic_summary.csv"
COEFFICIENTS_FILE = OUTPUT_DIR / "logistic_coefficients.csv"


def validate_market_features(df: pd.DataFrame) -> None:
    feature_sets = {"MARKET": MARKET_FEATURES,
                    "MARKET_Z60": MARKET_Z_FEATURES,
                    "MARKET_COMPACT": MARKET_COMPACT_FEATURES}

    for name, features in feature_sets.items():
        values = df[features].apply(pd.to_numeric, errors="coerce")

        missing = values.isna().sum()
        missing = missing[missing > 0]

        if not missing.empty:
            raise ValueError(f"Braki w {name}:\n{missing}")

        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"{name} zawiera wartości inf lub -inf")


def validate_sec_features(df: pd.DataFrame) -> None:
    for feature in SEC_BINARY_CANDIDATES:
        if df[feature].isna().any():
            raise ValueError(f"Cecha {feature} zawiera NaN")

        values = set(df[feature].unique())

        if not values.issubset({0, 1}):
            raise ValueError(f"Cecha {feature} nie jest binarna: {sorted(values)}")


def validate_sentiment(df: pd.DataFrame) -> None:
    if df["Mean_Net_Sentiment"].isna().any():
        raise ValueError("Brak Mean_Net_Sentiment")

    invalid_momentum = df["Sentiment_Momentum_3"].isna() & (df[SENTIMENT_HISTORY_FLAG] == 1)
    

    if invalid_momentum.any():
        rows = df.loc[invalid_momentum, ["Ticker", "Accession", "Sentiment_History_Count_3"]]

        raise ValueError(f"Brak Sentiment_Momentum_3 mimo dostępnej historii:\n{rows.to_string(index=False)}")



# Model - regresja logstyczna
def build_logistic_pipeline(numeric_features: list[str],
                            binary_features: list[str],
                            sentiment_features: list[str] | None = None) -> Pipeline:

    sentiment_features = sentiment_features or []
    transformers = []

    # StandardScaler jest trenowany tylko na train w danym foldzie
    if numeric_features:
        transformers.append(("numeric", StandardScaler(), numeric_features))

    transformers.append(("categorical",
                        OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
                        CATEGORICAL_FEATURES))

    if binary_features:
        transformers.append(("binary", "passthrough", binary_features))

    if sentiment_features:
        sentiment_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scaler", StandardScaler())
                ])

        transformers.append(("sentiment", sentiment_pipeline, sentiment_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    classifier = LogisticRegression(l1_ratio=0.0,
                                    C=1.0,
                                    solver="lbfgs",
                                    max_iter=5000,
                                    class_weight=None)

    return Pipeline(steps=[("preprocessor", preprocessor),
                            ("classifier", classifier),
        ])


# Pojedyńczy model - fold
def evaluate_logistic_model(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_features: list[str],
    binary_features: list[str],
    test_year: int,
    sentiment_features: list[str] | None = None
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:

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

    model = build_logistic_pipeline(numeric_features=numeric_features,
                                    binary_features=binary_features,
                                    sentiment_features=sentiment_features)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = calculate_metrics(y_true=y_test, y_pred=y_pred, y_prob=y_prob)

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    classifier = model.named_steps["classifier"]

    coefficients = pd.DataFrame({"Test_Year": test_year,
                                 "Model": model_name,
                                 "Feature": feature_names,
                                 "Coefficient": classifier.coef_[0]})

    coefficients["Abs_Coefficient"] = coefficients["Coefficient"].abs()

    predictions = test_df[
        ["Ticker", "Event_Session", "Accession", "Abnormal_Event_Return_1D"]].copy()

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
        **metrics
    }

    add_confusion_metrics(result=result, y_true=y_test, y_pred=y_pred,)

    return result, predictions, coefficients


# Model Dummy
def evaluate_dummy(train_df: pd.DataFrame, test_df: pd.DataFrame, test_year: int) -> tuple[dict, pd.DataFrame]:

    y_train = train_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)

    # DummyClassifier dostaje dowolnego X
    X_train = np.zeros((len(train_df), 1))
    X_test = np.zeros((len(test_df), 1))

    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    positive_index = int(np.where(model.classes_ == 1)[0][0])

    y_prob = model.predict_proba(X_test)[:, positive_index]

    result = {
        "Test_Year": test_year,
        "Model": "DUMMY",
        "Train_Size": len(train_df),
        "Test_Size": len(test_df),
        "Num_Input_Features": 0,
        "Num_Encoded_Features": 0,
        "Selected_SEC_Features": "",
        **calculate_metrics(y_test, y_pred, y_prob),
    }

    add_confusion_metrics(result=result, y_true=y_test, y_pred=y_pred)

    predictions = test_df[
        ["Ticker", "Event_Session", "Accession", "Abnormal_Event_Return_1D"]].copy()

    predictions["Test_Year"] = test_year
    predictions["Model"] = "DUMMY"
    predictions["y_true"] = y_test.to_numpy()
    predictions["y_pred"] = y_pred
    predictions["y_prob"] = y_prob

    return result, predictions


# Konfiguracja modeli
def build_model_configs(selected_sec: list[str]) -> list[dict]:
    return [{"name": "MODEL A - MARKET",
            "numeric": MARKET_FEATURES,
            "binary": [],
            "sentiment": []},

            {"name": "MODEL A2 - MARKET Z60",
            "numeric": MARKET_Z_FEATURES,
            "binary": [],
            "sentiment": []},

            {"name": "MODEL A3 - MARKET COMPACT",
            "numeric": MARKET_COMPACT_FEATURES,
            "binary": [],
            "sentiment": []},

            {"name": "MODEL B - MARKET + SEC",
            "numeric": MARKET_FEATURES,
            "binary": selected_sec,
            "sentiment": []},

            {"name": "MODEL B3 - COMPACT + SEC",
            "numeric": MARKET_COMPACT_FEATURES,
            "binary": selected_sec,
            "sentiment": []},

            {"name": "MODEL C - MARKET + SEC + FINBERT",
            "numeric": MARKET_FEATURES,
            "binary": selected_sec + [SENTIMENT_HISTORY_FLAG],
            "sentiment": SENTIMENT_FEATURES},

            {"name": "MODEL C2 - MARKET + SEC + FINBERT CONTEXT",
            "numeric": MARKET_FEATURES,
            "binary": selected_sec + [SENTIMENT_HISTORY_FLAG],
            "sentiment": SENTIMENT_CONTEXT_FEATURES},

            {"name": "MODEL C3 - COMPACT + SEC + FINBERT",
            "numeric": MARKET_COMPACT_FEATURES,
            "binary": selected_sec + [SENTIMENT_HISTORY_FLAG],
            "sentiment": SENTIMENT_FEATURES}]


# Main
def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f'Nie znaleziono pliku: {DATA_FILE}')

    df = pd.read_csv(DATA_FILE)
    df = prepare_model_dataset(df, TARGET)
    df = add_sentiment_context(df)

    validate_target(df, TARGET)
    validate_market_features(df)
    validate_sec_features(df)
    validate_sentiment(df)
    log_repeated_events(df)

    logger.info("Target: %s", TARGET)
    logger.info("Liczba obserwacji primary: %d", len(df))
    logger.info("Zakres danych: %s -> %s", df["Event_Session"].min().date(), df["Event_Session"].max().date())
    logger.info("Rozkład targetu:\n%s", df[TARGET].value_counts().sort_index().to_string())

    results = []
    all_predictions = []
    all_coefficients = []

    for test_year in TEST_YEARS:
        train_df = df[df["Event_Session"].dt.year < test_year].copy()

        test_df = df[df["Event_Session"].dt.year == test_year].copy()

        if train_df.empty or test_df.empty:
            logger.warning("Pominięcie foldu dla roku %d - pusty TRAIN lub TEST", test_year)
            continue

        if train_df[TARGET].nunique() != 2:
            raise ValueError(f'TRAIN przed {test_year} nie zawiera obu klas')

        logger.info("FOLD %d | TRAIN=%d (%s -> %s) | TEST=%d (%s -> %s)",
                    test_year,
                    len(train_df),
                    train_df["Event_Session"].min().date(),
                    train_df["Event_Session"].max().date(),
                    len(test_df),
                    test_df["Event_Session"].min().date(),
                    test_df["Event_Session"].max().date())

        logger.info("Target TRAIN:\n%s", train_df[TARGET].value_counts().sort_index().to_string(),)

        logger.info("Target TEST:\n%s", test_df[TARGET].value_counts().sort_index().to_string(),)

        # Dummy
        dummy_result, dummy_predictions = evaluate_dummy(train_df=train_df, test_df=test_df, test_year=test_year)
        results.append(dummy_result)
        all_predictions.append(dummy_predictions)

        logger.info("DUMMY | TEST %d | BA=%.4f | ROC-AUC=%.4f",
                     test_year,
                     dummy_result["Balanced_Accuracy"],
                    dummy_result["ROC_AUC"])


        # Sekcja SEC tylko dla TRIAN
        selected_sec = select_sec_features(train_df, candidates=SEC_BINARY_CANDIDATES, min_count=MIN_SEC_OCCURRENCES)

        sec_counts = {feature: int(train_df[feature].sum()) for feature in selected_sec}

        logger.info("SEC features dla TEST %d:\n%s",
                     test_year,
                     pd.Series(sec_counts).to_string())


        # Modele regresji logistyczej
        model_configs = build_model_configs(selected_sec=selected_sec)

        for config in model_configs:
            result, predictions, coefficients = evaluate_logistic_model(
                model_name=config["name"],
                train_df=train_df,
                test_df=test_df,
                numeric_features=config["numeric"],
                binary_features=config["binary"],
                sentiment_features=config["sentiment"],
                test_year=test_year
            )

            results.append(result)
            all_predictions.append(predictions)
            all_coefficients.append(coefficients)

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
        raise RuntimeError("Nie udało sie wytrenowac modelu")

    if not all_predictions:
        raise RuntimeError("Nie udało sie wygenerowac predykcji")

    results_df = pd.DataFrame(results)

    predictions_df = pd.concat(all_predictions, ignore_index=True,)

    coefficients_df = (pd.concat(all_coefficients, ignore_index=True)
                        if all_coefficients else pd.DataFrame())

    summary_df = create_summary(results_df=results_df, predictions_df=predictions_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(FOLD_METRICS_FILE, index=False)

    predictions_df.to_csv(PREDICTIONS_FILE, index=False,)

    summary_df.to_csv(SUMMARY_FILE, index=False)

    if not coefficients_df.empty:
        coefficients_df.to_csv(COEFFICIENTS_FILE, index=False)

    logger.info("Średnie wyniki walk-forward:\n%s", summary_df.to_string(index=False))

    logger.info("Metryki foldów: %s", FOLD_METRICS_FILE)
    logger.info("Predykcje OOS: %s", PREDICTIONS_FILE)
    logger.info("Podsumowanie: %s", SUMMARY_FILE)

    if not coefficients_df.empty:
        logger.info("Współczynniki regresji: %s", COEFFICIENTS_FILE)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    main()