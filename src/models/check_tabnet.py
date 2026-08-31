# Hiperparametry dla TabNet w latach 2020 - 2022

import logging
from pathlib import Path

import optuna
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from src.models.model_config import (MIN_SEC_COUNT, RANDOM_STATE, SEC_BINARY_CANDIDATES,
    SENTIMENT_FEATURES, SENTIMENT_HISTORY_FLAG, TARGET)

from src.models.model_utils import prepare_model_dataset, select_sec_features


from src.models.train_tabnet import (BATCH_SIZE, MAX_EPOCHS, PATIENCE, VIRTUAL_BATCH_SIZE,
    build_tabnet_model, prepare_xy, set_seed)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "model_dataset.csv"


def objective(trial, df):
    params = {"width": trial.suggest_categorical("width", [4, 8, 16]),
              "n_steps": trial.suggest_int("n_steps", 2, 4),
              "lr": trial.suggest_float("lr", 0.001, 0.02, log=True),
              "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True)}

    ba_scores, auc_scores = [], []

    for year in [2021, 2022]:
        train_df = df[df["Event_Session"].dt.year < year].copy()
        val_df = df[df["Event_Session"].dt.year == year].copy()

        selected_sec = select_sec_features(train_df,
                                           candidates=SEC_BINARY_CANDIDATES,
                                           min_count=MIN_SEC_COUNT)

        binary_features = selected_sec + [SENTIMENT_HISTORY_FLAG]

        X_train, X_val, _ = prepare_xy(train_df, val_df, binary_features, SENTIMENT_FEATURES)

        y_train = train_df[TARGET].to_numpy(dtype="int64")
        y_val = val_df[TARGET].to_numpy(dtype="int64")

        seed = RANDOM_STATE + year
        set_seed(seed)

        model = build_tabnet_model(seed=seed, **params)

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

        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        ba_scores.append(balanced_accuracy_score(y_val, y_pred))
        auc_scores.append(roc_auc_score(y_val, y_prob))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_ba = sum(ba_scores) / len(ba_scores)
    mean_auc = sum(auc_scores) / len(auc_scores)

    trial.set_user_attr("Balanced_Accuracy", mean_ba)
    trial.set_user_attr("ROC_AUC", mean_auc)

    return (mean_ba + mean_auc) / 2


def main() -> None:
    df = pd.read_csv(DATA_FILE)
    df = prepare_model_dataset(df, TARGET)
    df = df[df["Event_Session"].dt.year <= 2022].copy()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))

    study.optimize(lambda trial: objective(trial, df), n_trials=20)

    best = study.best_trial

    logger.info("Najlepsze parametry: %s", best.params)
    logger.info("Score: %.4f", best.value)
    logger.info("Balanced Accuracy: %.4f", best.user_attrs["Balanced_Accuracy"])
    logger.info("ROC-AUC: %.4f", best.user_attrs["ROC_AUC"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    main()