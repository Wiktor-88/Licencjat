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
from sklearn.impute import SimpleImputer


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
    # Relatywna siła spółki względem rynku
    "Stock_vs_QQQ_1D",
    "Stock_vs_QQQ_3D",
    "Stock_vs_QQQ_5D",

    # Stan spółki
    "Volatility_14D",
    "Relative_Volume_20D",
    "RSI_14",
    "Price_to_SMA20",
    "Intraday_Return",
    "Daily_Range",

    # Stan szerokiego rynku
    "QQQ_Log_Return_1D",
    "QQQ_Log_Return_3D",
    "QQQ_Log_Return_5D",
    "QQQ_Volatility_14D",
]



###### CECHY Z ROLLING Z-SCORE (60 dni) ######



MARKET_Z_FEATURES = [
    "Log_Return_1D_Z60",
    "Log_Return_3D_Z60",
    "Log_Return_5D_Z60",
    "Volatility_14D_Z60",
    "Relative_Volume_20D_Z60",
    "RSI_14",
    "Price_to_SMA20_Z60",
    "Intraday_Return_Z60",
    "Daily_Range_Z60",

    "QQQ_Log_Return_1D",
    "QQQ_Log_Return_3D",
    "QQQ_Log_Return_5D",
    "QQQ_Volatility_14D",
]



######## CECHY DLA MODELU A3 Z VIF ##########

MARKET_COMPACT_FEATURES = [
    # Krótki i szerszy horyzont zwrotu spółki
    "Log_Return_1D_Z60",
    "Log_Return_5D_Z60",

    # Stan / reżim spółki
    "Volatility_14D_Z60",
    "Relative_Volume_20D_Z60",
    "RSI_14",
    "Price_to_SMA20_Z60",
    "Intraday_Return_Z60",
    "Daily_Range_Z60",

    # Benchmark - krótki i szerszy horyzont
    "QQQ_Log_Return_1D",
    "QQQ_Log_Return_5D",
    "QQQ_Volatility_14D",
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
######## CECHY FINBERT ###########

SENTIMENT_FEATURES = [
    "Mean_Net_Sentiment",
    "Sentiment_Momentum_3",
]

SENTIMENT_CONTEXT_FEATURES = [
    "Mean_Net_Sentiment",
    "Sentiment_Momentum_3",
    "Abs_Sentiment",
    "Sentiment_x_Prior_Return_5D",
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
    sentiment_features: list[str] | None = None,
) -> Pipeline:

    if sentiment_features is None:
        sentiment_features = []


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
                (
                    "scaler",
                    StandardScaler(),
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
    sentiment_features=sentiment_features,
)

    model.fit(
        X_train,
        y_train,
    )

    if (
        model_name
        == "MODEL C - MARKET + SEC + FINBERT"
        and test_year == 2026
    ):

        preprocessor = model.named_steps[
            "preprocessor"
        ]

        classifier = model.named_steps[
            "classifier"
        ]

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        coefficients = pd.DataFrame(
            {
                "Feature": feature_names,
                "Coefficient":
                    classifier.coef_[0],
            }
        )

        coefficients["Abs_Coefficient"] = (
            coefficients["Coefficient"]
            .abs()
        )

        print(
            "\nMODEL C 2026 - "
            "współczynniki:"
        )

        print(
            coefficients
            .sort_values(
                "Abs_Coefficient",
                ascending=False,
            )
            .to_string(index=False)
        )




    y_pred = model.predict(
        X_test
    )

    y_prob = model.predict_proba(
        X_test
    )[:, 1]


    predictions = test_df[
        [
            "Ticker",
            "Event_Session",
            "Accession",
            "Abnormal_Event_Return_1D",
        ]
    ].copy()

    predictions["Test_Year"] = test_year
    predictions["Model"] = model_name
    predictions["y_true"] = y_test.to_numpy()
    predictions["y_pred"] = y_pred
    predictions["y_prob"] = y_prob

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
        predictions,
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
        + MARKET_Z_FEATURES
        + MARKET_COMPACT_FEATURES
        + CATEGORICAL_FEATURES
        + SEC_BINARY_CANDIDATES
        + SENTIMENT_FEATURES
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


    results = []
    all_predictions = []
    
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
        # MODEL A2
        # MARKET Z-SCORE + TICKER
        # ====================================================

        model_a2_result, model_a2_cm, model_a2_predictions = (
            evaluate_logistic_model(
                model_name="MODEL A2 - MARKET Z60",
                train_df=train_df,
                test_df=test_df,
                numeric_features=MARKET_Z_FEATURES,
                binary_features=[],
                test_year=test_year,
            )
        )

        results.append(
            model_a2_result
        )   





        # ====================================================
        # MODEL A3
        # COMPACT MARKET Z-SCORE + TICKER
        # ====================================================

        model_a3_result, model_a3_cm, model_a3_predictions = (
            evaluate_logistic_model(
                model_name="MODEL A3 - MARKET COMPACT",
                train_df=train_df,
                test_df=test_df,
                numeric_features=MARKET_COMPACT_FEATURES,
                binary_features=[],
                test_year=test_year,
            )
        )

        results.append(
            model_a3_result
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

        model_a_result, model_a_cm, model_a_predictions = (
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

        model_b_result, model_b_cm, model_b_predictions = (
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
        # DODATKOWE CECHY KONTEKSTOWE SENTYMENTU
        # ====================================================
        #
        # Wszystkie informacje są dostępne przed Event_Session:
        #
        # Mean_Net_Sentiment:
        # sentiment bieżącego komunikatu.
        #
        # Stock_vs_QQQ_5D:
        # wcześniejszy 5-dniowy ruch spółki względem QQQ,
        # liczony na Feature_Cutoff_Session.
        #
        # Brak sentimentu pozostaje NaN.
        # Później istniejący preprocessing sentimentu
        # imputuje go wartością 0.
        # ====================================================

        for frame in [
            train_df,
            test_df,
        ]:

            frame["Abs_Sentiment"] = (
                pd.to_numeric(
                    frame["Mean_Net_Sentiment"],
                    errors="coerce",
                )
                .abs()
            )

            frame["Sentiment_x_Prior_Return_5D"] = (
                pd.to_numeric(
                    frame["Mean_Net_Sentiment"],
                    errors="coerce",
                )
                *
                pd.to_numeric(
                    frame["Stock_vs_QQQ_5D"],
                    errors="coerce",
                )
            )



        # ====================================================
        # MODEL C2
        # MARKET + SEC + FINBERT + CONTEXT
        # ====================================================

        model_c2_result, model_c2_cm, model_c2_predictions = (
            evaluate_logistic_model(
                model_name="MODEL C2 - MARKET + SEC + FINBERT CONTEXT",
                train_df=train_df,
                test_df=test_df,
                numeric_features=MARKET_FEATURES,
                binary_features=selected_sec,
                sentiment_features=SENTIMENT_CONTEXT_FEATURES,
                test_year=test_year,
            )
        )

        results.append(
            model_c2_result
        )



        # ====================================================
        # MODEL C
        # MARKET + SEC + FINBERT
        # ====================================================

        model_c_binary_features = (
            selected_sec
        )

        model_c_result, model_c_cm, model_c_predictions = (
            evaluate_logistic_model(
                model_name="MODEL C - MARKET + SEC + FINBERT",
                train_df=train_df,
                test_df=test_df,
                numeric_features=MARKET_FEATURES,
                binary_features=model_c_binary_features,
                sentiment_features=SENTIMENT_FEATURES,
                test_year=test_year,
            )
        )

        results.append(
            model_c_result
        )





        ####################### INNE MODELE ####################

        # ====================================================
        # MODEL B3
        # MARKET COMPACT + SEC
        # ====================================================

        model_b3_result, model_b3_cm, model_b3_predictions = (
            evaluate_logistic_model(
                model_name="MODEL B3 - COMPACT + SEC",
                train_df=train_df,
                test_df=test_df,
                numeric_features=MARKET_COMPACT_FEATURES,
                binary_features=selected_sec,
                test_year=test_year,
            )
        )

        results.append(
            model_b3_result
        )





        # ====================================================
        # MODEL C3
        # MARKET COMPACT + SEC + FINBERT
        # ====================================================

        model_c3_result, model_c3_cm, model_c3_predictions = (
            evaluate_logistic_model(
                model_name="MODEL C3 - COMPACT + SEC + FINBERT",
                train_df=train_df,
                test_df=test_df,
                numeric_features=MARKET_COMPACT_FEATURES,
                binary_features=selected_sec,
                sentiment_features=SENTIMENT_FEATURES,
                test_year=test_year,
            )
        )

        results.append(
            model_c3_result
        )



        # PREDYKSZONS#
        all_predictions.append(model_a_predictions)
        all_predictions.append(model_a2_predictions)
        all_predictions.append(model_a3_predictions)
        all_predictions.append(model_b_predictions)
        all_predictions.append(model_b3_predictions)
        all_predictions.append(model_c_predictions)
        all_predictions.append(model_c2_predictions)
        all_predictions.append(model_c3_predictions)








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
                model_a2_result,
                model_a3_result,
                model_b_result,
                model_b3_result,
                model_c_result,
                model_c2_result,
                model_c3_result,
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

        print(
            "\nConfusion Matrix - Model A2:"
        )

        print(
            model_a2_cm
        )


        print(
            "\nConfusion Matrix - Model A3:"
        )

        print(
            model_a3_cm
        )

        print(
            "\nConfusion Matrix - Model C:"
        )

        print(
            model_c_cm
        )

        print(
            "\nConfusion Matrix - Model B3:"
        )

        print(
            model_b3_cm
        )

        print(
            "\nConfusion Matrix - Model C3:"
        )

        print(
            model_c3_cm
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

    # LOL# 

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    predictions_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "oos_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    print(
        "\nPredykcje OOS zapisano do:"
    )

    print(
        predictions_path
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