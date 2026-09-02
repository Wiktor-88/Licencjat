# Pełne XAI dla TabNetu

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.model_config import (CATEGORICAL_FEATURES, TABULAR_MARKET_FEATURES,
    MIN_SEC_COUNT, RANDOM_STATE, SEC_BINARY_CANDIDATES, TARGET, TEST_YEARS)

from src.models.model_utils import prepare_model_dataset, select_sec_features

from src.models.train_tabnet import (BATCH_SIZE, VIRTUAL_BATCH_SIZE,
    build_model_configs, build_tabnet_model, prepare_xy, set_seed)

from src.xai.common import save_dataframe, select_local_classification_examples

from src.xai.permutation_importance import (calculate_permutation_importance,
    plot_permutation_importance, summarize_permutation_importance)

from src.xai.tabnet import (TabNetPredictor, plot_tabnet_importance,
    run_global_tabnet_xai, run_local_tabnet_xai, summarize_tabnet_importance)


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "xai" / "tabnet"

DATA_FILE = DATA_DIR / "model_dataset.csv"
OOS_FILE = DATA_DIR / "tabnet_oos_predictions.csv"
METRICS_FILE = DATA_DIR / "tabnet_fold_metrics.csv"

XAI_MODELS = {"TABNET A - MARKET": "A",
              "TABNET B - MARKET + SEC": "B",
              "TABNET C - MARKET + SEC + FINBERT": "C"}


def get_saved_epochs(metrics_df: pd.DataFrame,
                     model_name: str,
                     test_year: int) -> int: 
    rows = metrics_df[(metrics_df["Model"] == model_name) & (metrics_df["Test_Year"] == test_year)]

    if len(rows) != 1:
        raise ValueError(f"{model_name} {test_year}: brak jednoznacznego Final_Epochs")

    epochs = int(rows.iloc[0]["Final_Epochs"])

    if epochs < 1:
        raise ValueError(f"{model_name} {test_year}: niepoprawne Final_Epochs")

    return epochs


def verify_oos_predictions(saved_oos: pd.DataFrame,
                           model_name: str,
                           test_year: int,
                           test_df: pd.DataFrame,
                           y_prob: np.ndarray,
                           y_pred: np.ndarray) -> None:
    expected = (saved_oos[(saved_oos["Model"] == model_name)
                          & (saved_oos["Test_Year"] == test_year)]
        .sort_values(["Ticker", "Event_Session", "Accession"]).reset_index(drop=True))

    current = test_df.copy()
    current["XAI_y_prob"] = y_prob
    current["XAI_y_pred"] = y_pred
    current = (current.sort_values(["Ticker", "Event_Session", "Accession"])
               .reset_index(drop=True))

    if len(expected) != len(current):
        raise ValueError(f"{model_name} {test_year}: różna liczba obserwacji OOS")

    if not np.array_equal(expected[["Ticker", "Accession"]].astype(str).to_numpy(),
                          current[["Ticker", "Accession"]].astype(str).to_numpy()):
        raise ValueError(f"{model_name} {test_year}: różne obserwacje OOS")

    if not np.allclose(expected["y_prob"].astype(float),
                       current["XAI_y_prob"].astype(float),
                       rtol=1e-6,
                       atol=1e-7):
        max_diff = float(np.max(np.abs(expected["y_prob"].astype(float).to_numpy()
            - current["XAI_y_prob"].astype(float).to_numpy())))

        raise ValueError(f"{model_name} {test_year}: y_prob niezgodne, max diff={max_diff:.6g}")

    if not np.array_equal(expected["y_pred"].astype(int),
                          current["XAI_y_pred"].astype(int)):
        raise ValueError(f"{model_name} {test_year}: y_pred nie zgadzają się z OOS")


def run_local_examples(c_models: dict, c_oos_rows: list[pd.DataFrame]) -> None:
    local_pool = pd.concat(c_oos_rows, ignore_index=True)

    examples = select_local_classification_examples(local_pool, "XAI_Probability", "XAI_Prediction")

    save_dataframe(examples[["XAI_Example_Type",
                            "Ticker",
                            "Accession",
                            "Event_Session",
                            "Test_Year",
                            TARGET,
                            "XAI_Prediction",
                            "XAI_Probability"]],
                   OUTPUT_DIR / "C" / "local_examples.csv")

    for _, example in examples.iterrows():
        test_year = int(example["Test_Year"])
        fold = c_models[test_year]

        event_df = local_pool[local_pool["Accession"].astype(str) == str(example["Accession"])].copy()

        if len(event_df) != 1:
            raise ValueError(f"Accession {example['Accession']} nie jest jednoznaczny")

        event_row = event_df.iloc[0].copy()
        event_row["XAI_Example_Type"] = example["XAI_Example_Type"]

        run_local_tabnet_xai(model=fold["model"],
                             preprocessor=fold["preprocessor"],
                             X_row=event_df[fold["features"]].copy(),
                             event_row=event_row,
                             xai_name="tabnet/C")

        logger.info("Local XAI | %s | TEST %d | p=%.4f",
                    example["XAI_Example_Type"],
                    test_year,
                    example["XAI_Probability"])


def main() -> None:
    set_seed(RANDOM_STATE)

    df = prepare_model_dataset(pd.read_csv(DATA_FILE), TARGET)
    saved_oos = pd.read_csv(OOS_FILE)
    saved_metrics = pd.read_csv(METRICS_FILE)

    all_native = []
    all_masks = []
    all_permutation = []

    c_models = {}
    c_oos_rows = []

    for test_year in TEST_YEARS:
        train_df = df[df["Event_Session"].dt.year < test_year].copy()
        test_df = df[df["Event_Session"].dt.year == test_year].copy()

        selected_sec = select_sec_features(train_df, SEC_BINARY_CANDIDATES, MIN_SEC_COUNT)

        configs = build_model_configs(selected_sec)

        for variant_id, config in enumerate(configs):
            variant = XAI_MODELS[config["name"]]
            seed = RANDOM_STATE + test_year + variant_id

            set_seed(seed)

            final_epochs = get_saved_epochs(saved_metrics, config["name"], test_year)

            input_features = list(dict.fromkeys(TABULAR_MARKET_FEATURES
                + CATEGORICAL_FEATURES + config["binary"] + config["sentiment"]))

            X_train, X_test, preprocessor = prepare_xy(train_df,
                                                       test_df,
                                                       config["binary"],
                                                       config["sentiment"])

            y_train = train_df[TARGET].to_numpy(dtype=np.int64)
            y_test = test_df[TARGET].to_numpy(dtype=np.int64)

            model = build_tabnet_model(seed)

            model.fit(X_train=X_train,
                      y_train=y_train,
                      max_epochs=final_epochs,
                      patience=0,
                      batch_size=BATCH_SIZE,
                      virtual_batch_size=VIRTUAL_BATCH_SIZE,
                      num_workers=0,
                      drop_last=False)

            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            verify_oos_predictions(saved_oos,
                                   config["name"],
                                   test_year,
                                   test_df,
                                   y_prob,
                                   y_pred)

            feature_names = preprocessor.get_feature_names_out().astype(str).tolist()
            

            native, masks = run_global_tabnet_xai(model,
                                                  X_test,
                                                  feature_names,
                                                  f"tabnet/{variant}/test_{test_year}")

            native.insert(0, "Test_Year", test_year)
            native.insert(0, "Variant", variant)
            all_native.append(native)

            masks.insert(0, "Test_Year", test_year)
            masks.insert(0, "Variant", variant)
            all_masks.append(masks)

            raw_X_test = test_df[input_features].copy()

            predictor = TabNetPredictor(model, preprocessor)

            permutation = calculate_permutation_importance(predictor,
                                                           raw_X_test,
                                                           pd.Series(y_test),
                                                           n_repeats=100,
                                                           random_state=RANDOM_STATE)

            permutation.insert(0, "Test_Year", test_year)
            permutation.insert(0, "Variant", variant)
            all_permutation.append(permutation)

            save_dataframe(permutation,
                OUTPUT_DIR / variant / f"test_{test_year}" / "permutation_importance.csv")

            logger.info("%s | TEST %d | epochs=%d | native=%d | masks=%d | PFI=%d | OOS zgodne",
                        config["name"],
                        test_year,
                        final_epochs,
                        len(native),
                        len(masks),
                        len(permutation))

            if variant == "C":
                c_models[test_year] = {"model": model,
                                       "preprocessor": preprocessor,
                                       "features": input_features}

                local_df = test_df.copy()
                local_df["Test_Year"] = test_year
                local_df["XAI_Probability"] = y_prob
                local_df["XAI_Prediction"] = y_pred
                c_oos_rows.append(local_df)

            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not all_native or not all_masks or not all_permutation:
        raise RuntimeError("Nie wygenerowano pełnych wyników XAI TabNet")

    native_df = pd.concat(all_native, ignore_index=True)
    masks_df = pd.concat(all_masks, ignore_index=True)
    pfi_df = pd.concat(all_permutation, ignore_index=True)

    native_stability = summarize_tabnet_importance(native_df, "Native_Importance")

    mask_stability = summarize_tabnet_importance(masks_df, "Mean_Explain_Weight")

    pfi_stability = summarize_permutation_importance(pfi_df)

    save_dataframe(native_df, OUTPUT_DIR / "native_importance_by_fold.csv")
    save_dataframe(native_stability, OUTPUT_DIR / "native_importance_stability.csv")
    save_dataframe(masks_df, OUTPUT_DIR / "mask_importance_by_fold.csv")
    save_dataframe(mask_stability, OUTPUT_DIR / "mask_importance_stability.csv")
    save_dataframe(pfi_df, OUTPUT_DIR / "permutation_importance_by_fold.csv")
    save_dataframe(pfi_stability, OUTPUT_DIR / "permutation_importance_stability.csv")

    for variant in ["A", "B", "C"]:
        native_variant = native_stability[native_stability["Variant"] == variant].copy()

        mask_variant = mask_stability[mask_stability["Variant"] == variant].copy()

        pfi_variant = pfi_stability[pfi_stability["Variant"] == variant].copy()

        plot_tabnet_importance(native_variant,
                               OUTPUT_DIR / f"native_importance_{variant}.png",
                               "Mean_Importance",
                              title=f"TabNet {variant} – natywna ważność")

        plot_tabnet_importance(mask_variant,
                               OUTPUT_DIR / f"mask_importance_{variant}.png",
                               "Mean_Importance",
                               title=f"TabNet {variant} – ważność masek OOS")

        plot_permutation_importance(pfi_variant,
                                    OUTPUT_DIR / f"permutation_importance_{variant}.png",
                                    metric="Mean_AUC_Drop")

    run_local_examples(c_models, c_oos_rows)

    logger.info("Zakończono XAI TabNet")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    main()
