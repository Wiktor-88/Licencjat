from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "decision_tree_oos_predictions.csv"
)


# ============================================================
# TARGET
# ============================================================

TARGET = "Target_Abnormal_1D"


# ============================================================
# CECHY
# ============================================================

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


CATEGORICAL_FEATURES = [
    "Ticker",
]


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


SENTIMENT_FEATURES = [
    "Mean_Net_Sentiment",
    "Sentiment_Momentum_3",
]


# ============================================================
# WALK-FORWARD
# ============================================================

TEST_YEARS = [
    2023,
    2024,
    2025,
    2026,
]


MIN_SEC_COUNT = 5


# ============================================================
# SEC FEATURE SELECTION
# ============================================================

def select_sec_features(
    train_df: pd.DataFrame,
) -> list[str]:

    selected = []

    for feature in SEC_BINARY_CANDIDATES:

        if feature == "Has_EX99":
            selected.append(feature)
            continue

        count = int(
            train_df[feature].sum()
        )

        if count >= MIN_SEC_COUNT:
            selected.append(feature)

    return selected


# ============================================================
# PIPELINE
# ============================================================

def build_tree_pipeline(
    numeric_features: list[str],
    binary_features: list[str],
    sentiment_features: list[str] | None = None,
) -> Pipeline:

    if sentiment_features is None:
        sentiment_features = []

    transformers = []

    # Market features:
    # brak skalowania
    transformers.append(
        (
            "numeric",
            "passthrough",
            numeric_features,
        )
    )

    # Ticker
    transformers.append(
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            CATEGORICAL_FEATURES,
        )
    )

    # SEC flags
    if binary_features:
        transformers.append(
            (
                "binary",
                "passthrough",
                binary_features,
            )
        )

    # FinBERT
    if sentiment_features:

        sentiment_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value=0.0,
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "sentiment",
                sentiment_pipeline,
                sentiment_features,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    classifier = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=8,
        random_state=42,
        class_weight=None,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


# ============================================================
# EWALUACJA
# ============================================================

def evaluate_tree(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_features: list[str],
    binary_features: list[str],
    test_year: int,
    sentiment_features: list[str] | None = None,
):

    if sentiment_features is None:
        sentiment_features = []

    input_features = (
        numeric_features
        + CATEGORICAL_FEATURES
        + binary_features
        + sentiment_features
    )

    X_train = train_df[
        input_features
    ].copy()

    X_test = test_df[
        input_features
    ].copy()

    y_train = train_df[
        TARGET
    ].astype(int)

    y_test = test_df[
        TARGET
    ].astype(int)

    model = build_tree_pipeline(
        numeric_features=numeric_features,
        binary_features=binary_features,
        sentiment_features=sentiment_features,
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

    result = {
        "Test_Year": test_year,
        "Model": model_name,
        "Train_Size": len(train_df),
        "Test_Size": len(test_df),
        "Num_Features_Raw": len(
            input_features
        ),
        "Accuracy": accuracy_score(
            y_test,
            y_pred,
        ),
        "Balanced_Accuracy":
            balanced_accuracy_score(
                y_test,
                y_pred,
            ),
        "Precision":
            precision_score(
                y_test,
                y_pred,
                zero_division=0,
            ),
        "Recall":
            recall_score(
                y_test,
                y_pred,
                zero_division=0,
            ),
        "F1":
            f1_score(
                y_test,
                y_pred,
                zero_division=0,
            ),
        "ROC_AUC":
            roc_auc_score(
                y_test,
                y_prob,
            ),
    }

    cm = confusion_matrix(
        y_test,
        y_pred,
    )

    predictions = test_df[
        [
            "Ticker",
            "Event_Session",
            "Accession",
            "Abnormal_Event_Return_1D",
        ]
    ].copy()

    predictions["Test_Year"] = (
        test_year
    )

    predictions["Model"] = (
        model_name
    )

    predictions["y_true"] = (
        y_test.to_numpy()
    )

    predictions["y_pred"] = (
        y_pred
    )

    predictions["y_prob"] = (
        y_prob
    )

    return (
        result,
        cm,
        predictions,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "DECISION TREE WALK-FORWARD"
    )

    print(
        "=" * 80
    )

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
    ).reset_index(
        drop=True
    )

    print(
        f"\nTarget: {TARGET}"
    )

    print(
        f"Liczba obserwacji: {len(df)}"
    )

    print(
        "\nRozkład targetu:"
    )

    print(
        df[TARGET].value_counts()
    )

    results = []
    all_predictions = []

    # ========================================================
    # WALK-FORWARD
    # ========================================================

    for test_year in TEST_YEARS:

        train_df = df[
            df["Event_Session"].dt.year
            < test_year
        ].copy()

        test_df = df[
            df["Event_Session"].dt.year
            == test_year
        ].copy()

        if test_df.empty:
            continue

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"TEST YEAR: {test_year}"
        )

        print(
            "=" * 80
        )

        print(
            f"Train size: {len(train_df)}"
        )

        print(
            f"Test size: {len(test_df)}"
        )

        selected_sec = (
            select_sec_features(
                train_df
            )
        )

        print(
            "\nSEC features:"
        )

        for feature in selected_sec:
            print(
                f"{feature}: "
                f"{int(train_df[feature].sum())}"
            )

        # ====================================================
        # TREE A
        # ====================================================

        tree_a_result, tree_a_cm, tree_a_predictions = (
            evaluate_tree(
                model_name="TREE A - COMPACT MARKET",
                train_df=train_df,
                test_df=test_df,
                numeric_features=MARKET_COMPACT_FEATURES,
                binary_features=[],
                test_year=test_year,
            )
        )

        # ====================================================
        # TREE B
        # ====================================================

        tree_b_result, tree_b_cm, tree_b_predictions = (
            evaluate_tree(
                model_name="TREE B - COMPACT + SEC",
                train_df=train_df,
                test_df=test_df,
                numeric_features=MARKET_COMPACT_FEATURES,
                binary_features=selected_sec,
                test_year=test_year,
            )
        )

        # ====================================================
        # TREE C
        # ====================================================

        tree_c_result, tree_c_cm, tree_c_predictions = (
            evaluate_tree(
                model_name="TREE C - COMPACT + SEC + FINBERT",
                train_df=train_df,
                test_df=test_df,
                numeric_features=MARKET_COMPACT_FEATURES,
                binary_features=selected_sec,
                sentiment_features=SENTIMENT_FEATURES,
                test_year=test_year,
            )
        )

        fold_results = pd.DataFrame(
            [
                tree_a_result,
                tree_b_result,
                tree_c_result,
            ]
        )

        print(
            "\nWyniki:"
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
            "\nConfusion Matrix - TREE A:"
        )

        print(
            tree_a_cm
        )

        print(
            "\nConfusion Matrix - TREE B:"
        )

        print(
            tree_b_cm
        )

        print(
            "\nConfusion Matrix - TREE C:"
        )

        print(
            tree_c_cm
        )

        results.extend(
            [
                tree_a_result,
                tree_b_result,
                tree_c_result,
            ]
        )

        all_predictions.extend(
            [
                tree_a_predictions,
                tree_b_predictions,
                tree_c_predictions,
            ]
        )

    # ========================================================
    # PODSUMOWANIE
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "ŚREDNIE PO FOLDACH"
    )

    print(
        "=" * 80
    )

    summary = (
        results_df
        .groupby("Model")
        [
            [
                "Accuracy",
                "Balanced_Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC_AUC",
            ]
        ]
        .mean()
        .sort_values(
            "Balanced_Accuracy",
            ascending=False,
        )
    )

    print(
        summary
    )

    # ========================================================
    # PREDYKCJE OOS
    # ========================================================

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    predictions_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nPredykcje zapisano do:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()