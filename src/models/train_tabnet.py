from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from pytorch_tabnet.tab_model import TabNetClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_dataset.csv"
)

OUTPUT_RESULTS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tabnet_walk_forward_results.csv"
)

OUTPUT_PREDICTIONS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tabnet_oos_predictions.csv"
)


# ============================================================
# SETTINGS
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

MAX_EPOCHS = 100
PATIENCE = 10

BATCH_SIZE = 256
VIRTUAL_BATCH_SIZE = 64


# ============================================================
# FEATURES
# ============================================================

MARKET_FEATURES = [
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

SENTIMENT_FEATURES = [
    "Mean_Net_Sentiment",
    "Sentiment_Momentum_3",
]


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# SEC FEATURES
# ============================================================

def select_sec_features(train_df):

    selected = []

    if "Has_EX99" in train_df.columns:
        selected.append("Has_EX99")

    item_columns = sorted(
        column
        for column in train_df.columns
        if column.startswith("Has_Item_")
    )

    for column in item_columns:

        count = int(
            train_df[column]
            .fillna(0)
            .sum()
        )

        if count >= MIN_SEC_COUNT:
            selected.append(column)

    return selected


# ============================================================
# PREPROCESSOR
# ============================================================

def fit_preprocessor(
    train_df,
    sec_features,
    include_sec,
    include_sentiment,
):

    ticker_categories = sorted(
        train_df["Ticker"]
        .astype(str)
        .unique()
    )

    market_scaler = StandardScaler()

    market_scaler.fit(
        train_df[MARKET_FEATURES]
        .astype(float)
    )

    sentiment_stats = {}

    if include_sentiment:

        for column in SENTIMENT_FEATURES:

            values = pd.to_numeric(
                train_df[column],
                errors="coerce",
            )

            mean = values.mean()
            std = values.std(ddof=0)

            if pd.isna(mean):
                mean = 0.0

            if pd.isna(std) or std < 1e-8:
                std = 1.0

            sentiment_stats[column] = (
                float(mean),
                float(std),
            )

    return {
        "ticker_categories":
            ticker_categories,

        "market_scaler":
            market_scaler,

        "sec_features":
            sec_features,

        "include_sec":
            include_sec,

        "include_sentiment":
            include_sentiment,

        "sentiment_stats":
            sentiment_stats,
    }


def transform_data(
    df,
    preprocessor,
):

    arrays = []

    feature_names = []

    # --------------------------------------------------------
    # MARKET
    # --------------------------------------------------------

    market = (
        preprocessor[
            "market_scaler"
        ]
        .transform(
            df[MARKET_FEATURES]
            .astype(float)
        )
        .astype(np.float32)
    )

    arrays.append(market)

    feature_names.extend(
        MARKET_FEATURES
    )

    # --------------------------------------------------------
    # TICKER ONE-HOT
    # --------------------------------------------------------

    categories = (
        preprocessor[
            "ticker_categories"
        ]
    )

    ticker_values = (
        df["Ticker"]
        .astype(str)
    )

    unknown = (
        set(ticker_values.unique())
        - set(categories)
    )

    if unknown:

        raise ValueError(
            "Ticker w validation/test "
            "niewidziany w train: "
            f"{sorted(unknown)}"
        )

    ticker_matrix = np.column_stack(
        [
            (
                ticker_values
                == ticker
            )
            .astype(np.float32)
            .to_numpy()
            for ticker in categories
        ]
    )

    arrays.append(
        ticker_matrix
    )

    feature_names.extend(
        [
            f"Ticker_{ticker}"
            for ticker in categories
        ]
    )

    # --------------------------------------------------------
    # SEC
    # --------------------------------------------------------

    if preprocessor[
        "include_sec"
    ]:

        sec_features = (
            preprocessor[
                "sec_features"
            ]
        )

        sec = (
            df[sec_features]
            .fillna(0)
            .astype(np.float32)
            .to_numpy()
        )

        arrays.append(sec)

        feature_names.extend(
            sec_features
        )

    # --------------------------------------------------------
    # FINBERT
    # --------------------------------------------------------

    if preprocessor[
        "include_sentiment"
    ]:

        sentiment_arrays = []

        for column in SENTIMENT_FEATURES:

            mean, std = (
                preprocessor[
                    "sentiment_stats"
                ][column]
            )

            values = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            standardized = (
                (values - mean)
                / std
            )

            # Brak EX99 pozostaje rozpoznawalny
            # przez Has_EX99 w wariancie C.
            standardized = (
                standardized
                .fillna(0.0)
            )

            sentiment_arrays.append(
                standardized
                .astype(np.float32)
                .to_numpy()
            )

        sentiment = np.column_stack(
            sentiment_arrays
        )

        arrays.append(sentiment)

        feature_names.extend(
            SENTIMENT_FEATURES
        )

    X = np.concatenate(
        arrays,
        axis=1,
    ).astype(np.float32)

    if not np.isfinite(X).all():

        raise ValueError(
            "NaN lub Inf po preprocessing."
        )

    return X, feature_names


# ============================================================
# MODEL
# ============================================================

def create_model(seed):

    return TabNetClassifier(

        n_d=8,
        n_a=8,
        n_steps=3,

        gamma=1.3,

        n_independent=2,
        n_shared=2,

        lambda_sparse=1e-4,

        optimizer_fn=torch.optim.Adam,

        optimizer_params={
            "lr": 0.01,
            "weight_decay": 1e-5,
        },

        mask_type="entmax",

        seed=seed,

        verbose=0,

        device_name=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_prob,
):

    y_pred = (
        y_prob >= 0.5
    ).astype(int)

    return {

        "Accuracy":
            accuracy_score(
                y_true,
                y_pred,
            ),

        "Balanced_Accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred,
            ),

        "Precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "Recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "F1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "ROC_AUC":
            roc_auc_score(
                y_true,
                y_prob,
            ),

        "y_pred":
            y_pred,

        "Confusion_Matrix":
            confusion_matrix(
                y_true,
                y_pred,
            ),
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    set_seed(
        RANDOM_STATE
    )

    print(
        "\nTABNET WALK-FORWARD"
    )

    print(
        "\nDevice:"
    )

    print(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    df = pd.read_csv(
        DATA_FILE
    )

    df["Event_Session"] = (
        pd.to_datetime(
            df["Event_Session"]
        )
    )

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
        "\nDataset:"
    )

    print(
        len(df)
    )

    print(
        "\nTarget:"
    )

    print(
        df[TARGET]
        .value_counts()
        .sort_index()
    )

    variants = [

        {
            "name":
                "TABNET A - MARKET",

            "include_sec":
                False,

            "include_sentiment":
                False,
        },

        {
            "name":
                "TABNET B - MARKET + SEC",

            "include_sec":
                True,

            "include_sentiment":
                False,
        },

        {
            "name":
                "TABNET C - MARKET + SEC + FINBERT",

            "include_sec":
                True,

            "include_sentiment":
                True,
        },
    ]

    all_results = []

    all_predictions = []

    # ========================================================
    # OUTER WALK-FORWARD
    # ========================================================

    for test_year in TEST_YEARS:

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

        train_df = df[
            df["Event_Session"]
            .dt.year
            < test_year
        ].copy()

        test_df = df[
            df["Event_Session"]
            .dt.year
            == test_year
        ].copy()

        validation_year = (
            test_year - 1
        )

        subtrain_df = train_df[
            train_df["Event_Session"]
            .dt.year
            < validation_year
        ].copy()

        val_df = train_df[
            train_df["Event_Session"]
            .dt.year
            == validation_year
        ].copy()

        print(
            "\nTrain:",
            len(train_df),
        )

        print(
            "Subtrain:",
            len(subtrain_df),
        )

        print(
            f"Validation {validation_year}:",
            len(val_df),
        )

        print(
            "Test:",
            len(test_df),
        )

        y_train = (
            train_df[TARGET]
            .to_numpy(
                dtype=np.int64
            )
        )

        y_subtrain = (
            subtrain_df[TARGET]
            .to_numpy(
                dtype=np.int64
            )
        )

        y_val = (
            val_df[TARGET]
            .to_numpy(
                dtype=np.int64
            )
        )

        y_test = (
            test_df[TARGET]
            .to_numpy(
                dtype=np.int64
            )
        )

        for variant_id, variant in enumerate(
            variants
        ):

            print(
                "\n"
                + "-" * 80
            )

            print(
                variant["name"]
            )

            # =================================================
            # INNER TRAIN / VALIDATION
            # =================================================

            inner_sec_features = (
                select_sec_features(
                    subtrain_df
                )
            )

            inner_preprocessor = (
                fit_preprocessor(
                    train_df=subtrain_df,
                    sec_features=(
                        inner_sec_features
                    ),
                    include_sec=(
                        variant[
                            "include_sec"
                        ]
                    ),
                    include_sentiment=(
                        variant[
                            "include_sentiment"
                        ]
                    ),
                )
            )

            X_subtrain, _ = (
                transform_data(
                    subtrain_df,
                    inner_preprocessor,
                )
            )

            X_val, _ = (
                transform_data(
                    val_df,
                    inner_preprocessor,
                )
            )

            seed = (
                RANDOM_STATE
                + test_year
                + variant_id
            )

            inner_model = (
                create_model(
                    seed
                )
            )

            inner_model.fit(

                X_train=X_subtrain,
                y_train=y_subtrain,

                eval_set=[
                    (
                        X_val,
                        y_val,
                    )
                ],

                eval_name=[
                    "validation"
                ],

                eval_metric=[
                    "logloss"
                ],

                max_epochs=MAX_EPOCHS,

                patience=PATIENCE,

                batch_size=BATCH_SIZE,

                virtual_batch_size=(
                    VIRTUAL_BATCH_SIZE
                ),

                num_workers=0,

                drop_last=False,
            )

            best_epoch = int(
                inner_model.best_epoch
            )

            # pytorch-tabnet liczy epoki od 0.
            final_epochs = (
                best_epoch + 1
            )

            best_val_loss = (
                float(
                    inner_model.best_cost
                )
            )

            print(
                "\nBest epoch:",
                best_epoch,
            )

            print(
                "Final epochs:",
                final_epochs,
            )

            print(
                "Best validation loss:",
                f"{best_val_loss:.6f}",
            )

            del inner_model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # =================================================
            # FINAL OUTER TRAIN PREPROCESSOR
            # =================================================

            sec_features = (
                select_sec_features(
                    train_df
                )
            )

            final_preprocessor = (
                fit_preprocessor(
                    train_df=train_df,
                    sec_features=(
                        sec_features
                    ),
                    include_sec=(
                        variant[
                            "include_sec"
                        ]
                    ),
                    include_sentiment=(
                        variant[
                            "include_sentiment"
                        ]
                    ),
                )
            )

            X_train, feature_names = (
                transform_data(
                    train_df,
                    final_preprocessor,
                )
            )

            X_test, _ = (
                transform_data(
                    test_df,
                    final_preprocessor,
                )
            )

            print(
                "Features:",
                X_train.shape[1],
            )

            # =================================================
            # FINAL TRAIN
            # =================================================

            final_model = (
                create_model(
                    seed
                )
            )

            final_model.fit(

                X_train=X_train,
                y_train=y_train,

                max_epochs=final_epochs,

                patience=0,

                batch_size=BATCH_SIZE,

                virtual_batch_size=(
                    VIRTUAL_BATCH_SIZE
                ),

                num_workers=0,

                drop_last=False,
            )

            # =================================================
            # TEST
            # =================================================

            y_prob = (
                final_model
                .predict_proba(
                    X_test
                )[:, 1]
            )

            metrics = (
                calculate_metrics(
                    y_true=y_test,
                    y_prob=y_prob,
                )
            )

            print(
                "\nAccuracy:",
                f"{metrics['Accuracy']:.6f}",
            )

            print(
                "Balanced Accuracy:",
                f"{metrics['Balanced_Accuracy']:.6f}",
            )

            print(
                "Precision:",
                f"{metrics['Precision']:.6f}",
            )

            print(
                "Recall:",
                f"{metrics['Recall']:.6f}",
            )

            print(
                "F1:",
                f"{metrics['F1']:.6f}",
            )

            print(
                "ROC-AUC:",
                f"{metrics['ROC_AUC']:.6f}",
            )

            print(
                "\nConfusion Matrix:"
            )

            print(
                metrics[
                    "Confusion_Matrix"
                ]
            )

            all_results.append(
                {
                    "Test_Year":
                        test_year,

                    "Validation_Year":
                        validation_year,

                    "Model":
                        variant["name"],

                    "Train_Size":
                        len(train_df),

                    "Validation_Size":
                        len(val_df),

                    "Test_Size":
                        len(test_df),

                    "Features":
                        X_train.shape[1],

                    "Best_Epoch":
                        best_epoch,

                    "Final_Epochs":
                        final_epochs,

                    "Best_Val_Loss":
                        best_val_loss,

                    "Accuracy":
                        metrics["Accuracy"],

                    "Balanced_Accuracy":
                        metrics[
                            "Balanced_Accuracy"
                        ],

                    "Precision":
                        metrics["Precision"],

                    "Recall":
                        metrics["Recall"],

                    "F1":
                        metrics["F1"],

                    "ROC_AUC":
                        metrics["ROC_AUC"],
                }
            )

            predictions = (
                test_df[
                    [
                        "Ticker",
                        "Accession",
                        "Event_Session",
                        "Abnormal_Event_Return_1D",
                    ]
                ]
                .copy()
            )

            predictions[
                "Test_Year"
            ] = test_year

            predictions[
                "Model"
            ] = variant["name"]

            predictions[
                "y_true"
            ] = y_test

            predictions[
                "y_pred"
            ] = metrics[
                "y_pred"
            ]

            predictions[
                "y_prob"
            ] = y_prob

            all_predictions.append(
                predictions
            )

            del final_model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ========================================================
    # RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    results_df.to_csv(
        OUTPUT_RESULTS,
        index=False,
    )

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
        "ŚREDNIE OOS"
    )

    print(
        "=" * 80
    )

    print(
        summary
    )

    predictions_df = (
        pd.concat(
            all_predictions,
            ignore_index=True,
        )
    )

    predictions_df.to_csv(
        OUTPUT_PREDICTIONS,
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
        "\nWyniki:"
    )

    print(
        OUTPUT_RESULTS
    )

    print(
        "\nPredykcje:"
    )

    print(
        OUTPUT_PREDICTIONS
    )