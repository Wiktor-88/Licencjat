# Ten plik jest dodatkowy i odpowiada za dostrajanie xgboosta


import logging
from pathlib import Path

import optuna
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from src.models.model_config import (CATEGORICAL_FEATURES, TABULAR_MARKET_FEATURES, MIN_SEC_COUNT,
    SEC_BINARY_CANDIDATES, SENTIMENT_FEATURES, SENTIMENT_HISTORY_FLAG, TARGET)
from src.models.model_utils import prepare_model_dataset, select_sec_features
from src.models.train_xgboost import build_preprocessor


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "model_dataset.csv"


def objective(trial, df):
    params = {"n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
              "max_depth": trial.suggest_int("max_depth", 1, 4),
              "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
              "min_child_weight": trial.suggest_int("min_child_weight", 2, 12)}

    ba_scores = []
    auc_scores = []

    for year in [2021, 2022]:
        train_df = df[df["Event_Session"].dt.year < year].copy()
        val_df = df[df["Event_Session"].dt.year == year].copy()

        selected_sec = select_sec_features(train_df,
                                           candidates=SEC_BINARY_CANDIDATES,
                                           min_count=MIN_SEC_COUNT)

        binary_features = selected_sec + [SENTIMENT_HISTORY_FLAG]
        features = TABULAR_MARKET_FEATURES + CATEGORICAL_FEATURES + binary_features + SENTIMENT_FEATURES
    

        preprocessor = build_preprocessor(numeric_features=TABULAR_MARKET_FEATURES,
                                          binary_features=binary_features,
                                          sentiment_features=SENTIMENT_FEATURES)

        X_train = preprocessor.fit_transform(train_df[features])
        X_val = preprocessor.transform(val_df[features])

        model = XGBClassifier(**params,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            reg_alpha=0.0,
                            reg_lambda=1.0,
                            objective="binary:logistic",
                            eval_metric="logloss",
                            tree_method="hist",
                            random_state=67,
                            n_jobs=-1)

        model.fit(X_train, train_df[TARGET])

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

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

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=67))

    study.optimize(lambda trial: objective(trial, df),
                    n_trials=30,)

    best = study.best_trial

    logger.info("Najlepsze parametry: %s", best.params)
    logger.info("Score: %.4f", best.value)
    logger.info("Balanced Accuracy: %.4f", best.user_attrs["Balanced_Accuracy"])
    logger.info("ROC-AUC: %.4f", best.user_attrs["ROC_AUC"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    main()
