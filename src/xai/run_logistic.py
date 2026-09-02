# Pełne XAI dla regresji logistycznej

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.model_config import (CATEGORICAL_FEATURES, MIN_SEC_COUNT,
    SEC_BINARY_CANDIDATES, TARGET, TEST_YEARS)
from src.models.model_utils import prepare_model_dataset, select_sec_features
from src.models.train_baseline import build_logistic_pipeline, build_model_configs
from src.xai.common import save_dataframe, select_local_classification_examples
from src.xai.logistic import extract_coefficients, plot_coefficients, run_local_logistic_xai


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "xai" / "logistic_regression"

DATA_FILE = DATA_DIR / "model_dataset.csv"
OOS_FILE = DATA_DIR / "oos_predictions.csv"

XAI_MODELS = {
    "MODEL A - MARKET": "A",
    "MODEL B - MARKET + SEC": "B",
    "MODEL C - MARKET + SEC + FINBERT": "C",
}



def verify_oos_predictions(saved_oos: pd.DataFrame,
                           model_name: str,
                           test_year: int,
                           test_df: pd.DataFrame,
                           y_prob: np.ndarray,
                           y_pred: np.ndarray) -> None:
    expected = (saved_oos[(saved_oos["Model"] == model_name) & (saved_oos["Test_Year"] == test_year)]
        .sort_values(["Ticker", "Event_Session", "Accession"]).reset_index(drop=True))

    current = test_df.copy()
    current["XAI_y_prob"] = y_prob
    current["XAI_y_pred"] = y_pred

    current = current.sort_values(["Ticker", "Event_Session", "Accession"]).reset_index(drop=True)
    

    if len(expected) != len(current):
        raise ValueError(f"{model_name} {test_year}: różna liczba obserwacji OOS")

    if not np.array_equal(
        expected[["Ticker", "Accession"]].astype(str).to_numpy(),
        current[["Ticker", "Accession"]].astype(str).to_numpy()):
        raise ValueError(f"{model_name} {test_year}: różne obserwacje OOS")

    if not np.allclose(expected["y_prob"].astype(float),
                       current["XAI_y_prob"].astype(float),
                       rtol=1e-7,
                       atol=1e-9):
        raise ValueError(f"{model_name} {test_year}: y_prob nie zgadzają się z OOS")

    if not np.array_equal(expected["y_pred"].astype(int),
                          current["XAI_y_pred"].astype(int)):
        raise ValueError(f"{model_name} {test_year}: y_pred nie zgadzają się z OOS")

def summarize_coefficients(df: pd.DataFrame) -> pd.DataFrame:
    result = (df.groupby(["Variant", "Feature"], as_index=False)
                .agg(
                    Mean_Coefficient=("Coefficient_Log_Odds", "mean"),
                    Mean_Abs_Coefficient=("Abs_Coefficient", "mean"),
                    Std_Coefficient=("Coefficient_Log_Odds", "std"),
                    Positive_Folds=("Coefficient_Log_Odds", lambda x: int((x > 0).sum())),
                    Negative_Folds=("Coefficient_Log_Odds", lambda x: int((x < 0).sum())),
                    N_Folds=("Test_Year", "nunique")))

    result["Sign_Consistency"] = (result[["Positive_Folds", "Negative_Folds"]].max(axis=1)
                                / result["N_Folds"])

    result["Presence_Rate"] = (result["N_Folds"] / len(TEST_YEARS))

    return (result.sort_values(["Variant", "Mean_Abs_Coefficient"], ascending=[True, False])
            .reset_index(drop=True))



def main() -> None:
    df = prepare_model_dataset(pd.read_csv(DATA_FILE), TARGET)

    c_models = {}
    c_oos_rows = []

    saved_oos = pd.read_csv(OOS_FILE)
    all_coefficients = []

    for test_year in TEST_YEARS:
        train_df = df[df["Event_Session"].dt.year < test_year].copy()
        test_df = df[df["Event_Session"].dt.year == test_year].copy()

        selected_sec = select_sec_features(train_df,
                                           SEC_BINARY_CANDIDATES,
                                           min_count=MIN_SEC_COUNT)

        configs = [config for config in build_model_configs(selected_sec)
                   if config["name"] in XAI_MODELS]

        for config in configs:
            input_features = list(dict.fromkeys(config["numeric"] + CATEGORICAL_FEATURES
                + config["binary"] + config["sentiment"]))

            X_train = train_df[input_features].copy()
            X_test = test_df[input_features].copy()
            y_train = train_df[TARGET].astype(int)

            model = build_logistic_pipeline(numeric_features=config["numeric"],
                                            binary_features=config["binary"],
                                            sentiment_features=config["sentiment"])

            model.fit(X_train, y_train)

            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test).astype(int)

            verify_oos_predictions(saved_oos=saved_oos,
                                   model_name=config["name"],
                                   test_year=test_year,
                                   test_df=test_df,
                                   y_prob=y_prob,
                                   y_pred=y_pred)

            if XAI_MODELS[config["name"]] == "C":
                c_models[test_year] = {"model": model, "features": input_features}

                local_df = test_df.copy()
                local_df["Test_Year"] = test_year
                local_df["XAI_Probability"] = y_prob
                local_df["XAI_Prediction"] = y_pred

                c_oos_rows.append(local_df)

            coefficients = extract_coefficients(model=model, X_reference=X_train)

            coefficients.insert(0, "Test_Year", test_year)
            coefficients.insert(0, "Variant", XAI_MODELS[config["name"]])

            all_coefficients.append(coefficients)

            logger.info("%s | TEST %d | cechy=%d | predykcje OOS zgodne",
                        config["name"],
                        test_year,
                        len(coefficients))

    coefficients_df = pd.concat(all_coefficients, ignore_index=True)

    stability_df = summarize_coefficients(coefficients_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_dataframe(coefficients_df, OUTPUT_DIR / "coefficients_by_fold.csv")

    save_dataframe(stability_df, OUTPUT_DIR / "coefficient_stability.csv")

    for variant in ["A", "B", "C"]:
        variant_df = (stability_df[stability_df["Variant"] == variant]
            .rename(columns={
                "Mean_Coefficient": "Coefficient_Log_Odds",
                "Mean_Abs_Coefficient": "Abs_Coefficient"}))

        plot_coefficients(variant_df, OUTPUT_DIR / f"coefficients_{variant}.png",
                          top_n=15,)

    logger.info("Zakończono globalne XAI regresji logistycznej")

    local_pool = pd.concat(c_oos_rows, ignore_index=True)

    examples = select_local_classification_examples(local_pool,
                                                    probability_column="XAI_Probability",
                                                    prediction_column="XAI_Prediction",)

    example_columns = ["XAI_Example_Type",
                       "Ticker",
                       "Accession",
                       "Event_Session",
                       "Test_Year",
                       TARGET,
                       "XAI_Prediction",
                       "XAI_Probability"]

    save_dataframe(examples[example_columns], OUTPUT_DIR / "C" / "local_examples.csv")

    for _, example in examples.iterrows():
        test_year = int(example["Test_Year"])
        fold = c_models[test_year]

        event_df = local_pool[local_pool["Accession"].astype(str) == str(example["Accession"])].copy()

        if len(event_df) != 1:
            raise ValueError(f"Accession {example['Accession']} nie jest jednoznaczny")

        event_row = event_df.iloc[0].copy()
        event_row["XAI_Example_Type"] = example["XAI_Example_Type"]

        run_local_logistic_xai(model=fold["model"],
                               X_row=event_df[fold["features"]].copy(),
                               event_row=event_row,
                               xai_name="logistic_regression/C")

        logger.info("Local XAI | %s | %s | TEST %d | p=%.4f",
                    example["XAI_Example_Type"],
                    example["Accession"],
                    test_year,
                    example["XAI_Probability"])

    logger.info("Zakończono local XAI regresji logistycznej")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    main()