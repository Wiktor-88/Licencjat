from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)

from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_dataset.csv"
)

TARGET = "Target_Abnormal_1D"


MARKET_COMPACT_FEATURES = [
    "Log_Return_1D_Z60",
    "Log_Return_5D_Z60",
    "Volatility_14D_Z60",
    "Relative_Volume_20D_Z60",
    "RSI_14",
    "Price_to_SMA20_Z60",
    "Intraday_Return_Z60",
    "Daily_Range_Z60",
    "QQQ_Log_Return_1D",
    "QQQ_Log_Return_5D",
    "QQQ_Volatility_14D",
]


SEC_FEATURES = [
    "Has_EX99",
    "Has_Item_2_02",
    "Has_Item_5_02",
    "Has_Item_5_03",
    "Has_Item_5_07",
    "Has_Item_7_01",
    "Has_Item_8_01",
]


FEATURES = (
    MARKET_COMPACT_FEATURES
    + ["Ticker"]
    + SEC_FEATURES
)


def calculate_metrics(
    name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
):

    model.fit(
        X_train,
        y_train,
    )

    train_pred = model.predict(
        X_train
    )

    train_prob = model.predict_proba(
        X_train
    )[:, 1]

    test_pred = model.predict(
        X_test
    )

    test_prob = model.predict_proba(
        X_test
    )[:, 1]

    print(
        "\n"
        + "=" * 70
    )

    print(name)

    print(
        "=" * 70
    )

    print("\nTRAIN:")

    print(
        "Accuracy:",
        accuracy_score(
            y_train,
            train_pred,
        ),
    )

    print(
        "Balanced Accuracy:",
        balanced_accuracy_score(
            y_train,
            train_pred,
        ),
    )

    print(
        "ROC-AUC:",
        roc_auc_score(
            y_train,
            train_prob,
        ),
    )

    print("\nTEST:")

    print(
        "Accuracy:",
        accuracy_score(
            y_test,
            test_pred,
        ),
    )

    print(
        "Balanced Accuracy:",
        balanced_accuracy_score(
            y_test,
            test_pred,
        ),
    )

    print(
        "ROC-AUC:",
        roc_auc_score(
            y_test,
            test_prob,
        ),
    )

    print(
        "\nProbability TRAIN:"
    )

    print(
        pd.Series(
            train_prob
        ).describe()
    )

    print(
        "\nProbability TEST:"
    )

    print(
        pd.Series(
            test_prob
        ).describe()
    )


def main():

    df = pd.read_csv(
        DATA_PATH
    )

    df["Event_Session"] = pd.to_datetime(
        df["Event_Session"]
    )

    df = df[
        (df["Use_In_Primary_Model"] == 1)
        & df[TARGET].notna()
    ].copy()

    df = df.sort_values(
        "Event_Session"
    )

    # ========================================================
    # SANITY CHECK TARGETU
    # ========================================================

    expected_target = (
        df["Abnormal_Event_Return_1D"] > 0
    ).astype(int)

    target_errors = (
        expected_target
        != df[TARGET].astype(int)
    ).sum()

    print(
        "Błędnie przypisane targety:",
        target_errors,
    )

    # ========================================================
    # TESTUJEMY FOLD 2025
    # ========================================================

    train_df = df[
        df["Event_Session"].dt.year
        < 2025
    ].copy()

    test_df = df[
        df["Event_Session"].dt.year
        == 2025
    ].copy()

    print(
        "\nTrain:",
        len(train_df),
    )

    print(
        "Test:",
        len(test_df),
    )

    print(
        "\nTarget TRAIN:"
    )

    print(
        train_df[TARGET]
        .value_counts()
        .sort_index()
    )

    print(
        "\nTarget TEST:"
    )

    print(
        test_df[TARGET]
        .value_counts()
        .sort_index()
    )

    # ========================================================
    # PREPROCESSING
    # ========================================================

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "market",
                "passthrough",
                MARKET_COMPACT_FEATURES,
            ),
            (
                "ticker",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                ["Ticker"],
            ),
            (
                "sec",
                "passthrough",
                SEC_FEATURES,
            ),
        ],
        remainder="drop",
    )

    X_train = preprocessor.fit_transform(
        train_df[FEATURES]
    )

    X_test = preprocessor.transform(
        test_df[FEATURES]
    )

    y_train = (
        train_df[TARGET]
        .astype(int)
    )

    y_test = (
        test_df[TARGET]
        .astype(int)
    )

    print(
        "\nShape TRAIN:",
        X_train.shape,
    )

    print(
        "Shape TEST:",
        X_test.shape,
    )

    # ========================================================
    # MODEL 1 - NASZE OBECNE PARAMETRY
    # ========================================================

    conservative_model = XGBClassifier(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.05,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    calculate_metrics(
        name="OBECNY XGBOOST",
        model=conservative_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    # ========================================================
    # MODEL 2 - TYLKO TEST ZDOLNOŚCI DOPASOWANIA
    # ========================================================
    #
    # Ten model NIE jest kandydatem finalnym.
    #
    # Cel: sprawdzić, czy pipeline/model w ogóle potrafi
    # nauczyć się danych treningowych.
    # ========================================================

    flexible_model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=1,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    calculate_metrics(
        name="FLEXIBLE XGBOOST - SANITY CHECK",
        model=flexible_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


if __name__ == "__main__":
    main()