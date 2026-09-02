import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.model_config import (CATEGORICAL_FEATURES, MIN_SEC_COUNT,
    RANDOM_STATE, SEC_BINARY_CANDIDATES, TARGET, TEST_YEARS)

from src.models.model_utils import select_sec_features

from src.models.train_lstm import (build_model_configs, fit_sequence_scaler,
    load_sequence_dataset, predict_model, prepare_metadata, prepare_static_features,
    set_seed, train_fixed_epochs, transform_sequences)

from src.xai.common import save_dataframe, select_local_classification_examples

from src.xai.sequence_ig import (plot_sequence_heatmap, plot_static_ig,
    run_global_sequence_ig,  run_local_sequence_ig, summarize_sequence_ig,
    summarize_static_ig)


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "xai" / "lstm"

SEQUENCE_FILE = DATA_DIR / "sequence_dataset.npz"
INDEX_FILE = DATA_DIR / "sequence_dataset_index.csv"
MODEL_DATASET_FILE = DATA_DIR / "model_dataset.csv"

OOS_FILE = DATA_DIR / "lstm_oos_predictions.csv"
METRICS_FILE = DATA_DIR / "lstm_fold_metrics.csv"

XAI_MODELS = {"LSTM A - MARKET SEQUENCE": "A",
              "LSTM B - MARKET SEQUENCE + SEC": "B",
              "LSTM C - MARKET SEQUENCE + SEC + FINBERT": "C"}


def get_saved_epochs(metrics_df: pd.DataFrame, model_name: str, test_year: int) -> int:
    rows = metrics_df[(metrics_df["Model"] == model_name) & (metrics_df["Test_Year"] == test_year)]

    if len(rows) != 1:
        raise ValueError(f"{model_name} {test_year}: brak jednoznacznego Final_Epochs")

    return int(rows.iloc[0]["Final_Epochs"])

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

def main() -> None:
    set_seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y, index_df, sequence_features = load_sequence_dataset(SEQUENCE_FILE, INDEX_FILE)

    metadata = prepare_metadata(index_df, pd.read_csv(MODEL_DATASET_FILE))

    saved_oos = pd.read_csv(OOS_FILE)
    saved_metrics = pd.read_csv(METRICS_FILE)

    years = metadata["Event_Session"].dt.year.to_numpy()

    all_sequence = []
    all_static = []

    c_models = {}
    c_oos_rows = []

    for test_year in TEST_YEARS:
        train_idx = np.where(years < test_year)[0]
        test_idx = np.where(years == test_year)[0]

        train_df = metadata.iloc[train_idx].copy()
        test_df = metadata.iloc[test_idx].copy()

        x_train_raw = X[train_idx]
        x_test_raw = X[test_idx]

        mean, std = fit_sequence_scaler(x_train_raw)
        x_train = transform_sequences(x_train_raw, mean, std)
        x_test = transform_sequences(x_test_raw, mean, std)

        y_train = y[train_idx]
        y_test = y[test_idx]

        selected_sec = select_sec_features(train_df,
                                           SEC_BINARY_CANDIDATES,
                                           MIN_SEC_COUNT)

        for variant_id, config in enumerate(build_model_configs(selected_sec)):
            variant = XAI_MODELS[config["name"]]
            seed = RANDOM_STATE + test_year + variant_id

            static_train, static_test, preprocessor = (prepare_static_features(
                train_df,
                test_df,
                config["binary"],
                config["sentiment"]))

            epochs = get_saved_epochs(saved_metrics,
                                      config["name"],
                                      test_year)

            model, _ = train_fixed_epochs(sequence_x=x_train,
                                          static_x=static_train,
                                          y=y_train,
                                          device=device,
                                          seed=seed,
                                          epochs=epochs)

            y_prob = predict_model(model, x_test, static_test, device)

            y_pred = (y_prob >= 0.5).astype(int)

            verify_oos_predictions(saved_oos,
                                   config["name"],
                                   test_year,
                                   test_df,
                                  y_prob,
                                   y_pred)

            static_features = preprocessor.get_feature_names_out().astype(str).tolist()
            

            sequence_df, static_df = run_global_sequence_ig(model=model,
                                                            sequence_x=x_test,
                                                            static_x=static_test,
                                                            sequence_features=sequence_features,
                                                            static_features=static_features,
                                                            device=device,
                                                            xai_name=f"lstm/{variant}/test_{test_year}",
                                                            n_steps=32)

            sequence_df.insert(0, "Test_Year", test_year)
            sequence_df.insert(0, "Variant", variant)
            all_sequence.append(sequence_df)

            static_df.insert(0, "Test_Year", test_year)
            static_df.insert(0, "Variant", variant)
            all_static.append(static_df)

            logger.info("%s | TEST %d | epochs=%d | OOS zgodne | IG gotowe",
                        config["name"],
                        test_year,
                        epochs)

            if variant == "C":
                c_models[test_year] = {"model": model,
                                       "preprocessor": preprocessor,
                                       "x_test": x_test,
                                       "static_test": static_test,
                                       "test_df": test_df.copy(),
                                       "static_features": static_features}

                local_df = test_df.copy()
                local_df["Test_Year"] = test_year
                local_df["XAI_Probability"] = y_prob
                local_df["XAI_Prediction"] = y_pred
                local_df["XAI_Fold_Position"] = np.arange(len(test_df))

                c_oos_rows.append(local_df)

    sequence_df = pd.concat(all_sequence, ignore_index=True)

    static_df = pd.concat(all_static, ignore_index=True)

    sequence_stability = summarize_sequence_ig(sequence_df)

    static_stability = summarize_static_ig(static_df)

    save_dataframe(sequence_df, OUTPUT_DIR / "sequence_ig_by_fold.csv")
    save_dataframe(sequence_stability, OUTPUT_DIR / "sequence_ig_stability.csv")
    save_dataframe(static_df, OUTPUT_DIR / "static_ig_by_fold.csv")
    save_dataframe(static_stability, OUTPUT_DIR / "static_ig_stability.csv")

    for variant in ["A", "B", "C"]:
        plot_sequence_heatmap(sequence_stability[sequence_stability["Variant"] == variant],
            OUTPUT_DIR / f"sequence_ig_{variant}.png")

        plot_static_ig(static_stability[static_stability["Variant"] == variant],
            OUTPUT_DIR / f"static_ig_{variant}.png")

    local_pool = pd.concat(c_oos_rows, ignore_index=True)

    examples = select_local_classification_examples(local_pool, "XAI_Probability", "XAI_Prediction")

    save_dataframe(
        examples[["XAI_Example_Type",
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

        row = fold["test_df"][fold["test_df"]["Accession"].astype(str) == str(example["Accession"])]

        if len(row) != 1:
            raise ValueError(f"Accession {example['Accession']} nie jest jednoznaczny")

        position = int(local_pool.loc[local_pool["Accession"].astype(str) == str(example["Accession"]),
                "XAI_Fold_Position"].iloc[0])

        event_row = row.iloc[0].copy()
        event_row["XAI_Example_Type"] = example["XAI_Example_Type"]

        run_local_sequence_ig(model=fold["model"],
                              sequence_x=fold["x_test"][position:position + 1],
                              static_x=fold["static_test"][position:position + 1],
                              sequence_features=sequence_features,
                              static_features=fold["static_features"],
                              event_row=event_row,
                              device=device,
                              xai_name="lstm/C",
                              n_steps=32)

    logger.info("Zakończono XAI LSTM")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    main()