# Plik odpowiedzialny za piąty model - Tabnet

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.model_config import (CATEGORICAL_FEATURES, TABULAR_MARKET_FEATURES, MIN_SEC_COUNT,
    RANDOM_STATE, SEC_BINARY_CANDIDATES, SENTIMENT_FEATURES, SENTIMENT_HISTORY_FLAG,
    TARGET, TEST_YEARS)
from src.models.model_utils import (add_confusion_metrics, calculate_metrics, create_summary,
    log_repeated_events, prepare_model_dataset, select_sec_features, validate_target)


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "model_dataset.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
FOLD_METRICS_FILE = OUTPUT_DIR / "tabnet_fold_metrics.csv"
PREDICTIONS_FILE = OUTPUT_DIR / "tabnet_oos_predictions.csv"
SUMMARY_FILE = OUTPUT_DIR / "tabnet_summary.csv"
IMPORTANCE_FILE = OUTPUT_DIR / "tabnet_feature_importance.csv"

MAX_EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 256
VIRTUAL_BATCH_SIZE = 64


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_preprocessor(binary_features: list[str],
                       sentiment_features: list[str] | None = None) -> ColumnTransformer:

    sentiment_features = sentiment_features or []

    transformers = [("market", StandardScaler(), TABULAR_MARKET_FEATURES),
                    ("ticker", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    CATEGORICAL_FEATURES)]

    if binary_features:
        transformers.append(("binary", "passthrough", binary_features))

    if sentiment_features:
        sentiment_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("sentiment", sentiment_pipe, sentiment_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_tabnet_model(seed: int, width: int = 16, n_steps: int = 4, lr: float = 0.005177401218093385,
    lambda_sparse: float = 5.87541610476001e-06) -> TabNetClassifier:

    return TabNetClassifier(n_d=width,
                            n_a=width,
                            n_steps=n_steps,
                            gamma=1.3,
                            n_independent=2,
                            n_shared=2,
                            lambda_sparse=lambda_sparse,
                            optimizer_fn=torch.optim.Adam,
                            optimizer_params={"lr": lr, "weight_decay": 1e-5},
                            mask_type="entmax",
                            seed=seed,
                            verbose=0,
                            device_name="cuda" if torch.cuda.is_available() else "cpu")


def build_model_configs(selected_sec: list[str]) -> list[dict]:
    return [{"name": "TABNET A - MARKET",
            "binary": [],
            "sentiment": []},

            {"name": "TABNET B - MARKET + SEC",
            "binary": selected_sec,
            "sentiment": []},

            {"name": "TABNET C - MARKET + SEC + FINBERT",
            "binary": selected_sec + [SENTIMENT_HISTORY_FLAG],
            "sentiment": SENTIMENT_FEATURES}]


def prepare_xy(train_df: pd.DataFrame,
               other_df: pd.DataFrame,
               binary_features: list[str],
               sentiment_features: list[str]):
    input_features = list(dict.fromkeys(TABULAR_MARKET_FEATURES + CATEGORICAL_FEATURES
                                        + binary_features + sentiment_features))

    preprocessor = build_preprocessor(binary_features, sentiment_features)

    X_train = preprocessor.fit_transform(train_df[input_features]).astype(np.float32)
    X_other = preprocessor.transform(other_df[input_features]).astype(np.float32)

    if not np.isfinite(X_train).all() or not np.isfinite(X_other).all():
        raise ValueError("NaN lub Inf po preprocessingu TabNet")

    return X_train, X_other, preprocessor


def find_best_epochs(subtrain_df: pd.DataFrame,
                    val_df: pd.DataFrame,
                    binary_features: list[str],
                    sentiment_features: list[str],
                    seed: int) -> tuple[int, float]:

    X_train, X_val, _ = prepare_xy(subtrain_df,
                                   val_df,
                                   binary_features,
                                   sentiment_features)

    y_train = subtrain_df[TARGET].to_numpy(dtype=np.int64)
    y_val = val_df[TARGET].to_numpy(dtype=np.int64)

    model = build_tabnet_model(seed)

    model.fit(X_train=X_train,
              y_train=y_train,
              eval_set=[(X_val, y_val)],
              eval_name=["validation"],
              eval_metric=["logloss"],
              max_epochs=MAX_EPOCHS,
              patience=PATIENCE,
              batch_size=BATCH_SIZE,
              virtual_batch_size=VIRTUAL_BATCH_SIZE,
              num_workers=0,
              drop_last=False)

    final_epochs = int(model.best_epoch) + 1
    best_loss = float(model.best_cost)

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_epochs, best_loss


def evaluate_tabnet(model_name: str,
                    train_df: pd.DataFrame,
                    test_df: pd.DataFrame,
                    binary_features: list[str],
                    sentiment_features: list[str],
                    final_epochs: int,
                    best_val_loss: float,
                    validation_year: int,
                    validation_size: int,
                    test_year: int,
                    seed: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:

    X_train, X_test, preprocessor = prepare_xy(train_df,
                                               test_df,
                                               binary_features,
                                               sentiment_features)

    y_train = train_df[TARGET].to_numpy(dtype=np.int64)
    y_test = test_df[TARGET].to_numpy(dtype=np.int64)

    model = build_tabnet_model(seed)

    model.fit(X_train=X_train,
              y_train=y_train,
              max_epochs=final_epochs,
              patience=0,
              batch_size=BATCH_SIZE,
              virtual_batch_size=VIRTUAL_BATCH_SIZE,
              num_workers=0,
              drop_last=False)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    feature_names = preprocessor.get_feature_names_out()

    importance = pd.DataFrame({"Test_Year": test_year,
                               "Model": model_name,
                               "Feature": feature_names,
                               "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)

    predictions = test_df[["Ticker", "Event_Session", "Accession", "Abnormal_Event_Return_1D",
        "Tradable_Abnormal_Return_1D"]].copy()

    predictions["Test_Year"] = test_year
    predictions["Model"] = model_name
    predictions["y_true"] = y_test
    predictions["y_pred"] = y_pred
    predictions["y_prob"] = y_prob

    result = {"Test_Year": test_year,
              "Validation_Year": validation_year,
              "Model": model_name,
              "Train_Size": len(train_df),
              "Validation_Size": validation_size,
              "Test_Size": len(test_df),
              "Num_Encoded_Features": X_train.shape[1],
              "Final_Epochs": final_epochs,
              "Best_Val_Loss": best_val_loss,
              "Selected_SEC_Features": "|".join(feature for feature in binary_features if feature in SEC_BINARY_CANDIDATES),
              **calculate_metrics(y_test, y_pred, y_prob)}

    add_confusion_metrics(result, y_test, y_pred)

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result, predictions, importance


def main() -> None:
    set_seed(RANDOM_STATE)

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    df = prepare_model_dataset(df, TARGET)

    validate_target(df, TARGET)
    log_repeated_events(df)

    logger.info("TabNet działa na: %s", "GPU" if torch.cuda.is_available() else "CPU")
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

        validation_year = test_year - 1
        subtrain_df = train_df[train_df["Event_Session"].dt.year < validation_year].copy()
        val_df = train_df[train_df["Event_Session"].dt.year == validation_year].copy()

        if train_df.empty or test_df.empty or subtrain_df.empty or val_df.empty:
            logger.warning("Pominięcie foldu %d - pusty zbiór", test_year)
            continue

        logger.info("FOLD %d | TRAIN=%d | SUBTRAIN=%d | VAL %d=%d | TEST=%d",
                    test_year,
                    len(train_df),
                    len(subtrain_df),
                    validation_year,
                    len(val_df),
                    len(test_df))

        inner_sec = select_sec_features(subtrain_df,
                                        candidates=SEC_BINARY_CANDIDATES,
                                        min_count=MIN_SEC_COUNT)

        final_sec = select_sec_features(train_df,
                                        candidates=SEC_BINARY_CANDIDATES,
                                        min_count=MIN_SEC_COUNT)

        inner_configs = build_model_configs(inner_sec)
        final_configs = build_model_configs(final_sec)

        for variant_id, (inner_config, final_config) in enumerate(zip(inner_configs, final_configs)):
            seed = RANDOM_STATE + test_year + variant_id
            set_seed(seed)

            final_epochs, best_val_loss = find_best_epochs(subtrain_df=subtrain_df,
                                                           val_df=val_df,
                                                           binary_features=inner_config["binary"],
                                                           sentiment_features=inner_config["sentiment"],
                                                           seed=seed)

            result, predictions, importance = evaluate_tabnet(model_name=final_config["name"],
                                                              train_df=train_df,
                                                              test_df=test_df,
                                                              binary_features=final_config["binary"],
                                                              sentiment_features=final_config["sentiment"],
                                                              final_epochs=final_epochs,
                                                              best_val_loss=best_val_loss,
                                                              validation_year=validation_year,
                                                              validation_size=len(val_df),
                                                              test_year=test_year,
                                                              seed=seed)

            results.append(result)
            all_predictions.append(predictions)
            all_importances.append(importance)

            logger.info("%s | TEST %d | epochs=%d | ACC=%.4f | BA=%.4f | F1=%.4f | AUC=%.4f",
                        final_config["name"],
                        test_year,
                        final_epochs,
                        result["Accuracy"],
                        result["Balanced_Accuracy"],
                        result["F1"],
                        result["ROC_AUC"])

            logger.info("%s | TEST %d | TN=%d FP=%d FN=%d TP=%d",
                        final_config["name"],
                        test_year,
                        result["TN"],
                        result["FP"],
                        result["FN"],
                        result["TP"])

    if not results:
        raise RuntimeError("Nie udało się wytrenować TabNet")

    results_df = pd.DataFrame(results)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    importance_df = pd.concat(all_importances, ignore_index=True)

    summary_df = create_summary(results_df, predictions_df)

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
