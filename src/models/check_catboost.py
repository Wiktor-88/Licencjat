# Plik dodatkowy do walidacji catboosta w okresie 2020-2022

import logging
from pathlib import Path

import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from src.models.model_config import (MARKET_COMPACT_FEATURES, MIN_SEC_COUNT, SEC_BINARY_CANDIDATES,
    SENTIMENT_FEATURES, SENTIMENT_HISTORY_FLAG, TARGET)
from src.models.model_utils import prepare_model_dataset, select_sec_features


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "model_dataset.csv"


def objective(trial, df):
    params = {"iterations": trial.suggest_int("iterations", 100, 400, step=50),
              "depth": trial.suggest_int("depth", 3, 6),
              "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
              "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2.0, 10.0)}

    ba_scores, auc_scores = [], []

    for year in [2021, 2022]:
        train_df = df[df["Event_Session"].dt.year < year].copy()
        val_df = df[df["Event_Session"].dt.year == year].copy()

        selected_sec = select_sec_features(train_df,
                                           candidates=SEC_BINARY_CANDIDATES,
                                           min_count=MIN_SEC_COUNT)

        features = ( ["Ticker"] + MARKET_COMPACT_FEATURES + selected_sec
            + [SENTIMENT_HISTORY_FLAG] + SENTIMENT_FEATURES)

        X_train = train_df[features].copy()
        X_val = val_df[features].copy()

        X_train["Ticker"] = X_train["Ticker"].astype(str)
        X_val["Ticker"] = X_val["Ticker"].astype(str)

        model = CatBoostClassifier(**params,
                                   loss_function="Logloss",
                                   random_seed=42,
                                   verbose=False,
                                   allow_writing_files=False,
                                   thread_count=-1)

        model.fit(X_train, train_df[TARGET], cat_features=["Ticker"])

        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        ba_scores.append(balanced_accuracy_score(val_df[TARGET], y_pred))
        auc_scores.append(roc_auc_score(val_df[TARGET], y_prob))

    mean_ba = sum(ba_scores) / len(ba_scores)
    mean_auc = sum(auc_scores) / len(auc_scores)

    trial.set_user_attr("Balanced_Accuracy", mean_ba)
    trial.set_user_attr("ROC_AUC", mean_auc)

    return (mean_ba + mean_auc) / 2


def main():
    df = pd.read_csv(DATA_FILE)
    df = prepare_model_dataset(df, TARGET)
    df = df[df["Event_Session"].dt.year <= 2022].copy()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=67),)

    study.optimize(lambda trial: objective(trial, df), n_trials=30)

    best = study.best_trial

    logger.info("Najlepsze parametry: %s", best.params)
    logger.info("Score: %.4f", best.value)
    logger.info("Balanced Accuracy: %.4f", best.user_attrs["Balanced_Accuracy"])
    logger.info("ROC-AUC: %.4f", best.user_attrs["ROC_AUC"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    main()