# Ten plik odpowiada ze trnowanie czartwego modelu - Catoboosta

import logging
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

from model_config import (MARKET_COMPACT_FEATURES, MIN_SEC_COUNT, SEC_BINARY_CANDIDATES,
    SENTIMENT_FEATURES, SENTIMENT_HISTORY_FLAG, TARGET, TEST_YEARS,)
from model_utils import (add_confusion_metrics, calculate_metrics, create_summary,
    log_repeated_events, prepare_model_dataset, select_sec_features, validate_target)


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "model_dataset.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTIONS_FILE = OUTPUT_DIR / "catboost_oos_predictions.csv"
FOLD_METRICS_FILE = OUTPUT_DIR / "catboost_fold_metrics.csv"
SUMMARY_FILE = OUTPUT_DIR / "catboost_summary.csv"
IMPORTANCE_FILE = OUTPUT_DIR / "catboost_feature_importance.csv"


def build_catboost_model() -> CatBoostClassifier:
    return CatBoostClassifier(iterations=150,
                            depth=4,
                            learning_rate=0.09067703270954058,
                            l2_leaf_reg=2.791075410796539,
                            loss_function="Logloss",
                            eval_metric="AUC",
                            random_seed=42,
                            verbose=False,
                            allow_writing_files=False,
                            thread_count=-1)

def evaluate_catboost(model_name: str,
                      train_df: pd.DataFrame,
                      test_df: pd.DataFrame,
                      feature_columns: list[str],
                      binary_features: list[str],
                      test_year: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:

    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()

    # CatBoost dostaje ticker bezpośrednio jako kategorię
    X_train["Ticker"] = X_train["Ticker"].astype(str)
    X_test["Ticker"] = X_test["Ticker"].astype(str)

    y_train = train_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)

    if y_train.nunique() != 2:
        raise ValueError(f"TRAIN dla testu {test_year} nie zawiera obu klas targetu")

    model = build_catboost_model()

    model.fit(X_train, y_train, cat_features=["Ticker"])

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    importance = pd.DataFrame({"Test_Year": test_year,
                                "Model": model_name,
                                "Feature": feature_columns,
                                "Importance": model.get_feature_importance()}).sort_values("Importance", ascending=False)

    predictions = test_df[["Ticker", "Event_Session", "Accession", "Abnormal_Event_Return_1D"]].copy()

    predictions["Test_Year"] = test_year
    predictions["Model"] = model_name
    predictions["y_true"] = y_test.to_numpy()
    predictions["y_pred"] = y_pred
    predictions["y_prob"] = y_prob

    result = {"Test_Year": test_year,
             "Model": model_name,
             "Train_Size": len(train_df),
             "Test_Size": len(test_df),
             "Num_Input_Features": len(feature_columns),
             "Num_Encoded_Features": len(feature_columns),
             "Selected_SEC_Features": "|".join(feature for feature in binary_features if feature in SEC_BINARY_CANDIDATES),
             **calculate_metrics(y_test, y_pred, y_prob)}

    add_confusion_metrics(result=result, y_true=y_test, y_pred=y_pred)

    return result, predictions, importance


def build_model_configs(selected_sec: list[str]) -> list[dict]:
    return [{"name": "CAT A - COMPACT MARKET",
            "features": ["Ticker"] + MARKET_COMPACT_FEATURES,
            "binary": []},

            {"name": "CAT B - COMPACT + SEC",
            "features": ["Ticker"] + MARKET_COMPACT_FEATURES + selected_sec,
            "binary": selected_sec},

            {"name": "CAT C - COMPACT + SEC + FINBERT",
            "features": (["Ticker"] + MARKET_COMPACT_FEATURES + selected_sec +
                          [SENTIMENT_HISTORY_FLAG] + SENTIMENT_FEATURES),
            "binary": selected_sec + [SENTIMENT_HISTORY_FLAG]}]



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
    logger.info("Rozkład targetu:\n%s", df[TARGET].value_counts().sort_index().to_string())

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

        logger.info("Target TRAIN:\n%s", train_df[TARGET].value_counts().sort_index().to_string())
        logger.info("Target TEST:\n%s", test_df[TARGET].value_counts().sort_index().to_string())

        selected_sec = select_sec_features(train_df,
                                           candidates=SEC_BINARY_CANDIDATES,
                                           min_count=MIN_SEC_COUNT)

        sec_counts = {feature: int(train_df[feature].sum()) for feature in selected_sec}

        logger.info("SEC features dla TEST %d:\n%s",
                    test_year,
                    pd.Series(sec_counts).to_string())

        for config in build_model_configs(selected_sec):
            result, predictions, importance = evaluate_catboost(model_name=config["name"],
                                                                train_df=train_df,
                                                                test_df=test_df,
                                                                feature_columns=config["features"],
                                                                binary_features=config["binary"],
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
        raise RuntimeError("Nie udało się wytrenować CatBoosta")

    results_df = pd.DataFrame(results)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    importance_df = pd.concat(all_importances, ignore_index=True)

    summary_df = create_summary(results_df=results_df, predictions_df=predictions_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(FOLD_METRICS_FILE, index=False)
    predictions_df.to_csv(PREDICTIONS_FILE, index=False)
    importance_df.to_csv(IMPORTANCE_FILE, index=False)
    summary_df.to_csv(SUMMARY_FILE, index=False)

    logger.info("Średnie wyniki walk-forward:\n%s", summary_df.to_string(index=False))

    logger.info("Metryki foldów: %s", FOLD_METRICS_FILE)
    logger.info("Predykcje OOS: %s", PREDICTIONS_FILE)
    logger.info("Feature importance: %s", IMPORTANCE_FILE)
    logger.info("Podsumowanie: %s", SUMMARY_FILE)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    main()