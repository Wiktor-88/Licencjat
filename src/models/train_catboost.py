from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# ŚCIEŻKI
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_dataset.csv"
)

OUTPUT_PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "catboost_oos_predictions.csv"
)


# ============================================================
# USTAWIENIA
# ============================================================

TARGET = "Target_Abnormal_1D"

TEST_YEARS = [
    2023,
    2024,
    2025,
    2026,
]

RANDOM_STATE = 42

MIN_SEC_COUNT = 5


# ============================================================
# CECHY RYNKOWE
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


# ============================================================
# FINBERT
# ============================================================

SENTIMENT_FEATURES = [
    "Mean_Net_Sentiment",
    "Sentiment_Momentum_3",
]


# ============================================================
# POMOCNICZE
# ============================================================

def select_sec_features(
    train_df: pd.DataFrame,
) -> list[str]:
    """
    Wybiera cechy strukturalne SEC wyłącznie
    na podstawie TRAIN.

    Has_EX99 zawsze zostaje.

    Has_Item_* zostaje, jeżeli występuje
    co najmniej MIN_SEC_COUNT razy w TRAIN.
    """

    selected = []

    if "Has_EX99" in train_df.columns:
        selected.append(
            "Has_EX99"
        )

    item_columns = sorted(
        [
            column
            for column in train_df.columns
            if column.startswith(
                "Has_Item_"
            )
        ]
    )

    for column in item_columns:

        count = int(
            train_df[column]
            .fillna(0)
            .sum()
        )

        if count >= MIN_SEC_COUNT:
            selected.append(
                column
            )

    return selected


def build_feature_list(
    sec_features: list[str],
    include_sec: bool,
    include_sentiment: bool,
) -> list[str]:

    features = [
        "Ticker",
        *MARKET_COMPACT_FEATURES,
    ]

    if include_sec:
        features.extend(
            sec_features
        )

    if include_sentiment:
        features.extend(
            SENTIMENT_FEATURES
        )

    return features


def prepare_features(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    CatBoost może obsługiwać NaN
    w cechach numerycznych natywnie.

    Ticker pozostaje kategorią tekstową.
    """

    x = df[
        feature_columns
    ].copy()

    x["Ticker"] = (
        x["Ticker"]
        .astype(str)
    )

    # Pozostałe kolumny jawnie numeryczne.
    numeric_columns = [
        column
        for column in feature_columns
        if column != "Ticker"
    ]

    for column in numeric_columns:
        x[column] = pd.to_numeric(
            x[column],
            errors="coerce",
        )

    return x


def create_model() -> CatBoostClassifier:
    """
    Konserwatywny CatBoost.

    Brak early stoppingu na foldzie TEST,
    żeby nie stroić modelu na przyszłości.
    """

    return CatBoostClassifier(
        iterations=250,
        depth=4,
        learning_rate=0.03,
        l2_leaf_reg=5.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def evaluate_model(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    test_year: int,
):
    """
    Trenuje jeden wariant CatBoost
    i zwraca:
    - metryki,
    - confusion matrix,
    - predykcje OOS.
    """

    x_train = prepare_features(
        train_df,
        feature_columns,
    )

    x_test = prepare_features(
        test_df,
        feature_columns,
    )

    y_train = (
        train_df[TARGET]
        .astype(int)
    )

    y_test = (
        test_df[TARGET]
        .astype(int)
    )

    model = create_model()

    ticker_index = (
        feature_columns.index(
            "Ticker"
        )
    )

    model.fit(
        x_train,
        y_train,
        cat_features=[
            ticker_index
        ],
    )

    y_prob = (
        model.predict_proba(
            x_test
        )[:, 1]
    )

    y_pred = (
        y_prob >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            y_pred,
        )
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    if y_test.nunique() == 2:
        roc_auc = roc_auc_score(
            y_test,
            y_prob,
        )
    else:
        roc_auc = np.nan

    cm = confusion_matrix(
        y_test,
        y_pred,
    )

    result = {
        "Test_Year": test_year,
        "Model": model_name,
        "Train_Size": len(train_df),
        "Test_Size": len(test_df),
        "Num_Features_Raw":
            len(feature_columns),
        "Accuracy":
            accuracy,
        "Balanced_Accuracy":
            balanced_accuracy,
        "Precision":
            precision,
        "Recall":
            recall,
        "F1":
            f1,
        "ROC_AUC":
            roc_auc,
    }

    prediction_columns = [
        "Ticker",
        "Event_Session",
        "Accession",
        "Abnormal_Event_Return_1D",
    ]

    predictions = (
        test_df[
            prediction_columns
        ]
        .copy()
    )

    predictions[
        "Test_Year"
    ] = test_year

    predictions[
        "Model"
    ] = model_name

    predictions[
        "y_true"
    ] = y_test.to_numpy()

    predictions[
        "y_pred"
    ] = y_pred

    predictions[
        "y_prob"
    ] = y_prob

    return (
        result,
        cm,
        predictions,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 80
    )

    print(
        "CATBOOST WALK-FORWARD"
    )

    print(
        "=" * 80
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku:\n"
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    df["Event_Session"] = (
        pd.to_datetime(
            df["Event_Session"]
        )
    )

    # ========================================================
    # GŁÓWNY ZBIÓR
    # ========================================================

    df = df[
        (
            df[
                "Use_In_Primary_Model"
            ]
            == 1
        )
        &
        (
            df[TARGET]
            .notna()
        )
    ].copy()

    df[TARGET] = (
        df[TARGET]
        .astype(int)
    )

    df = df.sort_values(
        [
            "Event_Session",
            "Ticker",
            "Accession",
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"\nTarget: {TARGET}"
    )

    print(
        "\nLiczba obserwacji:",
        len(df),
    )

    print(
        "\nRozkład targetu:"
    )

    print(
        df[TARGET]
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

    results = []
    all_predictions = []

    # ========================================================
    # WALK-FORWARD
    # ========================================================

    for test_year in TEST_YEARS:

        train_df = df[
            df[
                "Event_Session"
            ].dt.year
            < test_year
        ].copy()

        test_df = df[
            df[
                "Event_Session"
            ].dt.year
            == test_year
        ].copy()

        if train_df.empty:
            continue

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
            "\nTrain size:",
            len(train_df),
        )

        print(
            "Test size:",
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

        # ====================================================
        # SEC FEATURE SELECTION TYLKO NA TRAIN
        # ====================================================

        sec_features = (
            select_sec_features(
                train_df
            )
        )

        print(
            "\nSEC features:"
        )

        for feature in sec_features:

            print(
                f"{feature}: "
                f"{int(train_df[feature].fillna(0).sum())}"
            )

        # ====================================================
        # A
        # ====================================================

        features_a = (
            build_feature_list(
                sec_features=[],
                include_sec=False,
                include_sentiment=False,
            )
        )

        (
            result_a,
            cm_a,
            predictions_a,
        ) = evaluate_model(
            model_name=(
                "CAT A - COMPACT MARKET"
            ),
            train_df=train_df,
            test_df=test_df,
            feature_columns=features_a,
            test_year=test_year,
        )

        # ====================================================
        # B
        # ====================================================

        features_b = (
            build_feature_list(
                sec_features=sec_features,
                include_sec=True,
                include_sentiment=False,
            )
        )

        (
            result_b,
            cm_b,
            predictions_b,
        ) = evaluate_model(
            model_name=(
                "CAT B - COMPACT + SEC"
            ),
            train_df=train_df,
            test_df=test_df,
            feature_columns=features_b,
            test_year=test_year,
        )

        # ====================================================
        # C
        # ====================================================

        features_c = (
            build_feature_list(
                sec_features=sec_features,
                include_sec=True,
                include_sentiment=True,
            )
        )

        (
            result_c,
            cm_c,
            predictions_c,
        ) = evaluate_model(
            model_name=(
                "CAT C - COMPACT + SEC + FINBERT"
            ),
            train_df=train_df,
            test_df=test_df,
            feature_columns=features_c,
            test_year=test_year,
        )

        results.extend(
            [
                result_a,
                result_b,
                result_c,
            ]
        )

        all_predictions.extend(
            [
                predictions_a,
                predictions_b,
                predictions_c,
            ]
        )

        # ====================================================
        # RAPORT FOLDU
        # ====================================================

        fold_results = pd.DataFrame(
            [
                result_a,
                result_b,
                result_c,
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
                    "Precision",
                    "Recall",
                    "F1",
                    "ROC_AUC",
                ]
            ].to_string(
                index=False
            )
        )

        print(
            "\nConfusion Matrix - CAT A:"
        )
        print(
            cm_a
        )

        print(
            "\nConfusion Matrix - CAT B:"
        )
        print(
            cm_b
        )

        print(
            "\nConfusion Matrix - CAT C:"
        )
        print(
            cm_c
        )

    # ========================================================
    # WYNIKI WSZYSTKICH FOLDÓW
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "WSZYSTKIE FOLDY"
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
    # ŚREDNIE
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
            by="Balanced_Accuracy",
            ascending=False,
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "ŚREDNIE OOS"
    )

    print(
        "=" * 80
    )

    print(
        summary
    )

    # ========================================================
    # PREDYKCJE
    # ========================================================

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    OUTPUT_PREDICTIONS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_df.to_csv(
        OUTPUT_PREDICTIONS_FILE,
        index=False,
    )

    print(
        "\nLiczba predykcji OOS:"
    )

    print(
        len(predictions_df)
    )

    print(
        "\nPredykcje per model:"
    )

    print(
        predictions_df[
            "Model"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nPredykcje zapisano do:"
    )

    print(
        OUTPUT_PREDICTIONS_FILE
    )