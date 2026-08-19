from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# ŚCIEŻKI
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_dataset.csv"
)


# ============================================================
# TARGET
# ============================================================

TARGET_COLUMN = "Target_Abnormal_1D"


# ============================================================
# CECHY RYNKOWE
# ============================================================

MARKET_FEATURES = [
    "Log_Return_1D",
    "Log_Return_3D",
    "Log_Return_5D",
    "Volatility_14D",
    "Relative_Volume_20D",
    "RSI_14",
    "Price_to_SMA20",
    "Intraday_Return",
    "Daily_Range",
    "QQQ_Log_Return_1D",
    "QQQ_Log_Return_3D",
    "QQQ_Log_Return_5D",
    "QQQ_Volatility_14D",
    "Stock_vs_QQQ_1D",
    "Stock_vs_QQQ_3D",
    "Stock_vs_QQQ_5D",
]


# ============================================================
# CECHY KATEGORYCZNE
# ============================================================

CATEGORICAL_FEATURES = [
    "Ticker",
]


# ============================================================
# STRUKTURALNE CECHY SEC
# ============================================================

SEC_BINARY_CANDIDATES = [
    "Has_EX99",
    "Has_Item_1_01",
    "Has_Item_1_02",
    "Has_Item_1_05",
    "Has_Item_2_01",
    "Has_Item_2_02",
    "Has_Item_2_03",
    "Has_Item_5_02",
    "Has_Item_5_03",
    "Has_Item_5_07",
    "Has_Item_7_01",
    "Has_Item_8_01",
]


# Minimalna liczba wystąpień flagi w TRAIN.
MIN_SEC_OCCURRENCES = 5


# ============================================================
# FOLDY WALK-FORWARD
# ============================================================

TEST_YEARS = [
    2023,
    2024,
    2025,
    2026,
]


# ============================================================
# METRYKI
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_prob=None,
):
    metrics = {
        "Accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "Balanced_Accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "F1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }

    if (
        y_prob is not None
        and pd.Series(y_true).nunique() == 2
    ):
        metrics["ROC_AUC"] = roc_auc_score(
            y_true,
            y_prob,
        )
    else:
        metrics["ROC_AUC"] = np.nan

    return metrics


# ============================================================
# SELEKCJA FLAG SEC
# ============================================================

def select_sec_features(
    train_df: pd.DataFrame,
) -> list[str]:
    """
    Zachowuje Has_EX99 oraz tylko te flagi Item,
    które wystąpiły co najmniej MIN_SEC_OCCURRENCES
    razy w aktualnym zbiorze treningowym.

    Selekcja jest wykonywana osobno w każdym foldzie,
    więc nie zaglądamy do przyszłości.
    """

    selected = []

    for feature in SEC_BINARY_CANDIDATES:

        count = int(
            train_df[feature].sum()
        )

        if feature == "Has_EX99":
            selected.append(feature)

        elif count >= MIN_SEC_OCCURRENCES:
            selected.append(feature)

    return selected


# ============================================================
# LOGISTIC REGRESSION PIPELINE
# ============================================================

def build_logistic_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    binary_features: list[str],
) -> Pipeline:

    transformers = []

    # Cechy ciągłe:
    # skalowanie Z-score liczone WYŁĄCZNIE na TRAIN.
    if numeric_features:
        transformers.append(
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            )
        )

    # Ticker:
    # one-hot encoding.
    if categorical_features:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
                categorical_features,
            )
        )

    # Flagi SEC:
    # pozostają dokładnie 0/1.
    if binary_features:
        transformers.append(
            (
                "binary",
                "passthrough",
                binary_features,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                    class_weight=None,
                ),
            ),
        ]
    )

    return model


# ============================================================
# POJEDYNCZY MODEL / FOLD
# ============================================================

def evaluate_logistic_model(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_features: list[str],
    binary_features: list[str],
    test_year: int,
):

    input_features = (
        numeric_features
        + CATEGORICAL_FEATURES
        + binary_features
    )

    X_train = train_df[
        input_features
    ].copy()

    y_train = train_df[
        TARGET_COLUMN
    ].astype(int)

    X_test = test_df[
        input_features
    ].copy()

    y_test = test_df[
        TARGET_COLUMN
    ].astype(int)

    model = build_logistic_pipeline(
        numeric_features=numeric_features,
        categorical_features=CATEGORICAL_FEATURES,
        binary_features=binary_features,
    )

    model.fit(
        X_train,
        y_train,
    )

    y_pred = model.predict(
        X_test
    )

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    metrics = calculate_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
    )

    result = {
        "Test_Year": test_year,
        "Model": model_name,
        "Train_Size": len(train_df),
        "Test_Size": len(test_df),
        "Num_Features_Raw": len(input_features),
        **metrics,
    }

    return (
        result,
        confusion_matrix(
            y_test,
            y_pred,
        ),
    )


# ============================================================
# DUMMY
# ============================================================

def evaluate_dummy(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_year: int,
):

    # Dummy potrzebuje dowolnego X.
    X_train = train_df[
        ["Log_Return_1D"]
    ]

    X_test = test_df[
        ["Log_Return_1D"]
    ]

    y_train = train_df[
        TARGET_COLUMN
    ].astype(int)

    y_test = test_df[
        TARGET_COLUMN
    ].astype(int)

    model = DummyClassifier(
        strategy="most_frequent",
    )

    model.fit(
        X_train,
        y_train,
    )

    y_pred = model.predict(
        X_test
    )

    metrics = calculate_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=None,
    )

    return {
        "Test_Year": test_year,
        "Model": "DUMMY",
        "Train_Size": len(train_df),
        "Test_Size": len(test_df),
        "Num_Features_Raw": 0,
        **metrics,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "WALK-FORWARD BASELINES"
    )

    print(
        "=" * 80
    )

    print(
        f"\nTarget: {TARGET_COLUMN}"
    )

    # ========================================================
    # WCZYTANIE
    # ========================================================

    df = pd.read_csv(
        DATA_PATH
    )

    df["Event_Session"] = pd.to_datetime(
        df["Event_Session"]
    )

    # ========================================================
    # PRIMARY MODEL ONLY
    # ========================================================

    df = df[
        (df["Use_In_Primary_Model"] == 1)
        & df[TARGET_COLUMN].notna()
    ].copy()

    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype(int)
    )

    df.sort_values(
        by=[
            "Event_Session",
            "Ticker",
            "Accession",
        ],
        inplace=True,
        ignore_index=True,
    )

    print(
        f"\nLiczba obserwacji: {len(df)}"
    )

    print(
        "\nRozkład targetu:"
    )

    print(
        df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    print(
        "\nZakres danych:"
    )

    print(
        df["Event_Session"].min(),
        "->",
        df["Event_Session"].max(),
    )

    # ========================================================
    # KONTROLA BRAKÓW DLA MODELI A/B
    # ========================================================

    required_features = (
        MARKET_FEATURES
        + CATEGORICAL_FEATURES
        + SEC_BINARY_CANDIDATES
    )

    missing_columns = [
        column
        for column in required_features
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Brak wymaganych kolumn:\n"
            + "\n".join(missing_columns)
        )

    market_nan = (
        df[MARKET_FEATURES]
        .isna()
        .sum()
    )

    if market_nan.any():

        raise ValueError(
            "Model A/B ma braki w cechach "
            "rynkowych:\n"
            f"{market_nan[market_nan > 0]}"
        )

    # ========================================================
    # WALK-FORWARD
    # ========================================================

    results = []

    for test_year in TEST_YEARS:

        train_df = df[
            df["Event_Session"].dt.year
            < test_year
        ].copy()

        test_df = df[
            df["Event_Session"].dt.year
            == test_year
        ].copy()

        if train_df.empty or test_df.empty:
            continue

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"FOLD: TEST {test_year}"
        )

        print(
            "=" * 80
        )

        print(
            "\nTRAIN:"
        )

        print(
            train_df["Event_Session"].min(),
            "->",
            train_df["Event_Session"].max(),
        )

        print(
            f"Liczba: {len(train_df)}"
        )

        print(
            "\nTarget TRAIN:"
        )

        print(
            train_df[TARGET_COLUMN]
            .value_counts()
            .sort_index()
        )

        print(
            "\nTEST:"
        )

        print(
            test_df["Event_Session"].min(),
            "->",
            test_df["Event_Session"].max(),
        )

        print(
            f"Liczba: {len(test_df)}"
        )

        print(
            "\nTarget TEST:"
        )

        print(
            test_df[TARGET_COLUMN]
            .value_counts()
            .sort_index()
        )

        # ====================================================
        # DUMMY
        # ====================================================

        dummy_result = evaluate_dummy(
            train_df=train_df,
            test_df=test_df,
            test_year=test_year,
        )

        results.append(
            dummy_result
        )

        # ====================================================
        # MODEL A
        # MARKET + TICKER
        # ====================================================

        model_a_result, model_a_cm = (
            evaluate_logistic_model(
                model_name="MODEL A - MARKET",
                train_df=train_df,
                test_df=test_df,
                numeric_features=MARKET_FEATURES,
                binary_features=[],
                test_year=test_year,
            )
        )

        results.append(
            model_a_result
        )

        # ====================================================
        # MODEL B
        # MARKET + SEC + TICKER
        # ====================================================

        selected_sec = select_sec_features(
            train_df
        )

        print(
            "\nSEC features użyte w tym foldzie:"
        )

        for feature in selected_sec:
            count = int(
                train_df[feature].sum()
            )

            print(
                f"  {feature}: {count}"
            )

        model_b_result, model_b_cm = (
            evaluate_logistic_model(
                model_name="MODEL B - MARKET + SEC",
                train_df=train_df,
                test_df=test_df,
                numeric_features=MARKET_FEATURES,
                binary_features=selected_sec,
                test_year=test_year,
            )
        )

        results.append(
            model_b_result
        )

        # ====================================================
        # PODGLĄD FOLDU
        # ====================================================

        print(
            "\nWyniki:"
        )

        fold_results = pd.DataFrame(
            [
                dummy_result,
                model_a_result,
                model_b_result,
            ]
        )

        print(
            fold_results[
                [
                    "Model",
                    "Accuracy",
                    "Balanced_Accuracy",
                    "F1",
                    "ROC_AUC",
                ]
            ].to_string(
                index=False
            )
        )

        print(
            "\nConfusion Matrix - Model A:"
        )

        print(
            model_a_cm
        )

        print(
            "\nConfusion Matrix - Model B:"
        )

        print(
            model_b_cm
        )

    # ========================================================
    # WSZYSTKIE FOLDY
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "WSZYSTKIE WYNIKI"
    )

    print(
        "=" * 80
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    # ========================================================
    # ŚREDNIE WALK-FORWARD
    # ========================================================

    metric_columns = [
        "Accuracy",
        "Balanced_Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
    ]

    summary = (
        results_df
        .groupby(
            "Model"
        )[metric_columns]
        .mean()
        .sort_values(
            "Balanced_Accuracy",
            ascending=False,
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "ŚREDNIE WALK-FORWARD"
    )

    print(
        "=" * 80
    )

    print(
        summary.to_string()
    )


if __name__ == "__main__":
    main()