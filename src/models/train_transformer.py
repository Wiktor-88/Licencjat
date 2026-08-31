# Plik do trenowania ostatniego modelu - transformera

import logging

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.train_lstm import (BATCH_SIZE, DROPOUT, LEARNING_RATE, MAX_EPOCHS,
    MIN_DELTA, PATIENCE, WEIGHT_DECAY,
    build_model_configs, calculate_loss, create_loader, load_sequence_dataset,
    predict_model, prepare_metadata, prepare_sequence_features, prepare_static_features,
    set_seed, train_one_epoch)

from src.models.model_config import (MIN_SEC_COUNT, RANDOM_STATE, SEC_BINARY_CANDIDATES,
    TARGET, TEST_YEARS)
from src.models.model_utils import (add_confusion_metrics, calculate_metrics,
    create_summary, log_repeated_events, select_sec_features)



logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

SEQUENCE_FILE = DATA_DIR / "sequence_dataset.npz"
INDEX_FILE = DATA_DIR / "sequence_dataset_index.csv"
MODEL_DATASET_FILE = DATA_DIR / "model_dataset.csv"

FOLD_METRICS_FILE = DATA_DIR / "transformer_fold_metrics.csv"
PREDICTIONS_FILE = DATA_DIR / "transformer_oos_predictions.csv"
SUMMARY_FILE = DATA_DIR / "transformer_summary.csv"


# Parametry dla transformera
D_MODEL = 32
N_HEADS = 4
NUM_LAYERS = 1
DIM_FEEDFORWARD = 64


# Model
class TransformerClassifier(nn.Module):
    def __init__(self,
                 sequence_input_size: int,
                 static_input_size: int,
                 sequence_length: int,
                 d_model: int = D_MODEL,
                 n_heads: int = N_HEADS,
                 num_layers: int = NUM_LAYERS,
                 dim_feedforward: int = DIM_FEEDFORWARD,
                 dropout: float = DROPOUT):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model musi być podzielne przez n_heads")

        self.input_projection = nn.Linear(sequence_input_size, d_model)

        # Token zbierający informację z całej sekwencji
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # 30 sesji + token CLS
        self.position_embedding = nn.Parameter(torch.zeros(1, sequence_length + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=n_heads,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation="gelu",
                                                   batch_first=True)

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.final_norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(d_model + static_input_size, 32),
                                  nn.ReLU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(32, 1))

        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, sequence_x, static_x):
        # [batch, 30, 11] -> [batch, 30, d_model]
        x = self.input_projection(sequence_x)

        cls = self.cls_token.expand(x.shape[0], -1, -1,)

        x = torch.cat([cls, x], dim=1)

        x = x + self.position_embedding[:, :x.shape[1], :]

        x = self.transformer(x)

        # CLS jako reprezentacja całych 30 sesji
        sequence_representation = self.final_norm(x[:, 0, :])

        # Cechy statyczne dodajemy dopiero po Transformerze
        x = torch.cat([sequence_representation, static_x], dim=1)

        return self.head(x).squeeze(1)


# Budowa modelu
def build_transformer_model(sequence_input_size: int,
                            static_input_size: int,
                            sequence_length: int,
                            d_model: int = D_MODEL,
                            n_heads: int = N_HEADS,
                            num_layers: int = NUM_LAYERS,
                            dim_feedforward: int = DIM_FEEDFORWARD,
                            dropout: float = DROPOUT) -> TransformerClassifier:

    return TransformerClassifier(sequence_input_size=sequence_input_size,
                                 static_input_size=static_input_size,
                                 sequence_length=sequence_length,
                                 d_model=d_model,
                                 n_heads=n_heads,
                                 num_layers=num_layers,
                                 dim_feedforward=dim_feedforward,
                                 dropout=dropout)


# Liczba epok
def select_best_epoch(sequence_train: np.ndarray,
                      static_train: np.ndarray,
                      y_train: np.ndarray,
                      sequence_val: np.ndarray,
                      static_val: np.ndarray,
                      y_val: np.ndarray,
                      device: torch.device,
                      seed: int) -> tuple[int, float]:

    set_seed(seed)

    model = build_transformer_model(sequence_input_size=sequence_train.shape[2],
                                    static_input_size=static_train.shape[1],
                                    sequence_length=sequence_train.shape[1]).to(device)

    loader = create_loader(sequence_x=sequence_train,
                           static_x=static_train,
                           y=y_train,
                           batch_size=BATCH_SIZE,
                           shuffle=True)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)

    best_epoch = 1
    best_val_loss = np.inf
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model=model,
                                     loader=loader,
                                     optimizer=optimizer,
                                     criterion=criterion,
                                     device=device)

        val_loss = calculate_loss(model=model,
                                  sequence_x=sequence_val,
                                  static_x=static_val,
                                  y=y_val,
                                  device=device)

        logger.debug("Epoch %d | train=%.6f | val=%.6f", epoch, train_loss, val_loss)

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            break

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_epoch, float(best_val_loss)


# Trenowanie
def train_fixed_epochs(sequence_x: np.ndarray,
                       static_x: np.ndarray,
                       y: np.ndarray,
                       device: torch.device,
                       seed: int,
                       epochs: int) -> tuple[TransformerClassifier, float]:

    set_seed(seed)

    model = build_transformer_model(sequence_input_size=sequence_x.shape[2],
                                    static_input_size=static_x.shape[1],
                                    sequence_length=sequence_x.shape[1]).to(device)

    loader = create_loader(sequence_x=sequence_x,
                           static_x=static_x,
                           y=y,
                           batch_size=BATCH_SIZE,
                           shuffle=True)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)

    final_loss = np.nan

    for _ in range(epochs):
        final_loss = train_one_epoch(model=model,
                                     loader=loader,
                                     optimizer=optimizer,
                                     criterion=criterion,
                                     device=device)

    return model, float(final_loss)


def main() -> None:
    for file in [SEQUENCE_FILE, INDEX_FILE, MODEL_DATASET_FILE]:
        if not file.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku: {file}")

    set_seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Transformer działa na: %s", "GPU" if device.type == "cuda" else "CPU")

    X, y, index_df, feature_names = load_sequence_dataset(SEQUENCE_FILE, INDEX_FILE)

    model_df = pd.read_csv(MODEL_DATASET_FILE)

    metadata = prepare_metadata(index_df=index_df, model_df=model_df)

    if len(metadata) != len(X):
        raise ValueError("Metadata i tensor sekwencji mają różną liczbę obserwacji")

    if not np.array_equal(metadata[TARGET].astype(int).to_numpy(), y):
        raise ValueError('Target metadata nie zgadza się z targetem sekwencj')

    log_repeated_events(metadata)

    logger.info("Target: %s", TARGET)
    logger.info("Liczba obserwacji: %d", len(metadata))
    logger.info("Shape sekwencji: %s", X.shape)
    logger.info("Liczba cech sekwencyjnych: %d", len(feature_names))
    logger.info("Cechy sekwencyjne: %s", feature_names)

    results = []
    all_predictions = []

    years = metadata["Event_Session"].dt.year.to_numpy()

    # Walk forward
    for test_year in TEST_YEARS:
        train_idx = np.where(years < test_year)[0]
        test_idx = np.where(years == test_year)[0]

        if len(train_idx) == 0 or len(test_idx) == 0:
            logger.warning("Pominięcie foldu %d - pusty TRAIN lub TEST", test_year)
            continue

        train_df = metadata.iloc[train_idx].copy()
        test_df = metadata.iloc[test_idx].copy()

        x_train_raw = X[train_idx]
        x_test_raw = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        # Poprzedni rok wybiera liczbę epok
        validation_year = test_year - 1

        train_years = train_df["Event_Session"].dt.year.to_numpy()
        

        subtrain_pos = np.where(train_years < validation_year)[0]

        val_pos = np.where(train_years == validation_year)[0]

        if len(subtrain_pos) == 0 or len(val_pos) == 0:
            raise ValueError(f"Brak temporal validation dla testu {test_year}")

        subtrain_df = train_df.iloc[subtrain_pos].copy()

        val_df = train_df.iloc[val_pos].copy()

        x_subtrain_raw = x_train_raw[subtrain_pos]

        x_val_raw = x_train_raw[val_pos]

        y_subtrain = y_train[subtrain_pos]

        y_val = y_train[val_pos]

        logger.info("FOLD %d | TRAIN=%d | SUBTRAIN=%d | VAL %d=%d | TEST=%d",
                    test_year,
                    len(train_df),
                    len(subtrain_df),
                    validation_year,
                    len(val_df),
                    len(test_df))

        # Skalery są dopasowane wyłącznie na danych treningowych
        x_subtrain, x_val = prepare_sequence_features(x_subtrain_raw, x_val_raw)

        x_train, x_test = prepare_sequence_features(x_train_raw, x_test_raw)

        inner_sec = select_sec_features(subtrain_df,
                                        candidates=SEC_BINARY_CANDIDATES,
                                        min_count=MIN_SEC_COUNT)

        final_sec = select_sec_features(train_df,
                                        candidates=SEC_BINARY_CANDIDATES,
                                        min_count=MIN_SEC_COUNT)
                                    

        inner_configs = build_model_configs(inner_sec, model_name="TRANSFORMER")

        final_configs = build_model_configs(final_sec, model_name="TRANSFORMER")


        # Modele A,B i C
        for variant_id, (inner_config, final_config) in enumerate(zip(inner_configs, final_configs)):
            model_name = final_config["name"]

            seed = RANDOM_STATE + test_year + variant_id
            

            # Wewnętrzna walidacja
            static_subtrain, static_val, _ = prepare_static_features(
                train_df=subtrain_df,
                other_df=val_df,
                binary_features=inner_config["binary"],
                sentiment_features=inner_config["sentiment"])

            best_epoch, best_val_loss = select_best_epoch(sequence_train=x_subtrain,
                                                          static_train=static_subtrain,
                                                          y_train=y_subtrain,
                                                          sequence_val=x_val,
                                                          static_val=static_val,
                                                          y_val=y_val,
                                                          device=device,
                                                          seed=seed)

            # Końcowe trenowanie
            static_train, static_test, _ = prepare_static_features(
                train_df=train_df,
                other_df=test_df,
                binary_features=final_config["binary"],
                sentiment_features=final_config["sentiment"])

            model, final_train_loss = train_fixed_epochs(sequence_x=x_train,
                                                         static_x=static_train,
                                                         y=y_train,
                                                         device=device,
                                                         seed=seed,
                                                         epochs=best_epoch)

            
            # Wyniki na TRAIN - diagnostyka przeuczenia
            train_prob = predict_model(model=model,
                                       sequence_x=x_train,
                                       static_x=static_train,
                                       device=device)

            train_pred = (train_prob >= 0.5).astype(int)

            train_metrics = calculate_metrics(y_true=y_train,
                                              y_pred=train_pred,
                                              y_prob=train_prob)

            # test
            y_prob = predict_model(model=model,
                                   sequence_x=x_test,
                                   static_x=static_test,
                                   device=device)

            y_pred = (y_prob >= 0.5).astype(int)

            test_metrics = calculate_metrics(y_true=y_test,
                                             y_pred=y_pred, 
                                             y_prob=y_prob)

            result = {"Test_Year": test_year,
                      "Model": model_name,
                      "Train_Size": len(train_df),
                      "Validation_Year": validation_year,
                      "Validation_Size": len(val_df),
                      "Test_Size": len(test_df),

                      "Sequence_Length": X.shape[1],
                      "Sequence_Features": X.shape[2],
                      "Static_Features": static_train.shape[1],

                      "Final_Epochs": best_epoch,
                      "Best_Val_Loss": best_val_loss,
                      "Final_Train_Loss": final_train_loss,
 
                      "Train_Accuracy": train_metrics["Accuracy"],
                      "Train_Balanced_Accuracy": train_metrics["Balanced_Accuracy"],
                      "Train_Precision": train_metrics["Precision"],
                      "Train_Recall": train_metrics["Recall"],
                      "Train_F1": train_metrics["F1"],
                      "Train_ROC_AUC": train_metrics["ROC_AUC"],

                      "BA_Gap": (train_metrics["Balanced_Accuracy"]  - test_metrics["Balanced_Accuracy"]),
                      "AUC_Gap": (train_metrics["ROC_AUC"] - test_metrics["ROC_AUC"]),

                      "Selected_SEC_Features": "|".join(feature for feature in final_config["binary"]
                                                        if feature in SEC_BINARY_CANDIDATES),

                      **test_metrics}

            add_confusion_metrics(result=result, y_true=y_test, y_pred=y_pred)

            results.append(result)

            predictions = test_df[
                ["Ticker", "Event_Session", "Accession", "Abnormal_Event_Return_1D"]].copy()

            predictions["Test_Year"] = test_year
            predictions["Model"] = model_name
            predictions["y_true"] = y_test
            predictions["y_pred"] = y_pred
            predictions["y_prob"] = y_prob

            all_predictions.append(predictions)

            logger.info("%s | TEST %d | epochs=%d | ACC=%.4f | "
                        "BA=%.4f | F1=%.4f | AUC=%.4f",
                        model_name,
                        test_year,
                        best_epoch,
                        result["Accuracy"],
                        result["Balanced_Accuracy"],
                        result["F1"],
                        result["ROC_AUC"])

            logger.info("%s | TRAIN vs TEST | BA=%.4f -> %.4f | AUC=%.4f -> %.4f | "
                        "BA gap=%.4f | AUC gap=%.4f",
                        model_name,
                        train_metrics["Balanced_Accuracy"],
                        test_metrics["Balanced_Accuracy"],
                        train_metrics["ROC_AUC"],
                        test_metrics["ROC_AUC"],
                        result["BA_Gap"],
                        result["AUC_Gap"])

            logger.info("%s | TEST %d | TN=%d FP=%d FN=%d TP=%d",
                        model_name,
                        test_year,
                        result["TN"],
                        result["FP"],
                        result["FN"],
                        result["TP"])

            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    # Podsumowanie
    if not results:
        raise RuntimeError("Nie udało się wytrenować modeli Transformer")

    results_df = pd.DataFrame(results)

    predictions_df = pd.concat(all_predictions, ignore_index=True)

    summary_df = create_summary(results_df=results_df, predictions_df=predictions_df)

    results_df.to_csv(FOLD_METRICS_FILE, index=False)

    predictions_df.to_csv(PREDICTIONS_FILE, index=False)

    summary_df.to_csv(SUMMARY_FILE, index=False)

    logger.info("Średnie wyniki walk-forward:\n%s",
                summary_df.to_string(index=False))

    logger.info("Metryki foldów: %s", FOLD_METRICS_FILE)

    logger.info("Predykcje OOS: %s", PREDICTIONS_FILE)

    logger.info("Podsumowanie: %s", SUMMARY_FILE)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    main()