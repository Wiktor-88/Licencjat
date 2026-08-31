# Trenowanie sieci neuronowej - LSTM

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.models.model_config import (CATEGORICAL_FEATURES, MIN_SEC_COUNT, RANDOM_STATE,
    SEC_BINARY_CANDIDATES, SENTIMENT_FEATURES, SENTIMENT_HISTORY_FLAG, TARGET, TEST_YEARS)
from src.models.model_utils import (add_confusion_metrics, calculate_metrics, create_summary,
    log_repeated_events, prepare_model_dataset, select_sec_features, validate_target)


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

SEQUENCE_FILE = DATA_DIR / "sequence_dataset.npz"
INDEX_FILE = DATA_DIR / "sequence_dataset_index.csv"
MODEL_DATASET_FILE = DATA_DIR / "model_dataset.csv"

FOLD_METRICS_FILE = DATA_DIR / "lstm_fold_metrics.csv"
PREDICTIONS_FILE = DATA_DIR / "lstm_oos_predictions.csv"
SUMMARY_FILE = DATA_DIR / "lstm_summary.csv"


MAX_EPOCHS = 80
PATIENCE = 8
MIN_DELTA = 1e-4

BATCH_SIZE = 64
HIDDEN_SIZE = 32
DROPOUT = 0.20

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Tworzenie modelu
class LSTMClassifier(nn.Module):
    def __init__(self,
                 sequence_input_size: int,
                 static_input_size: int,
                 hidden_size: int = HIDDEN_SIZE,
                 dropout: float = DROPOUT):
        super().__init__()

        self.lstm = nn.LSTM(input_size=sequence_input_size,
                            hidden_size=hidden_size,
                            batch_first=True)

        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(hidden_size + static_input_size, 32),
                                  nn.ReLU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(32, 1))

    def forward(self, sequence_x, static_x):
        _, (hidden, _) = self.lstm(sequence_x)

        # Ostatni stan LSTM opisuje całą sekwencję
        sequence_representation = hidden[-1]

        x = torch.cat([sequence_representation, static_x], dim=1)

        return self.head(x).squeeze(1)


def build_lstm_model(sequence_input_size: int,
                     static_input_size: int,
                     hidden_size: int = HIDDEN_SIZE,
                     dropout: float = DROPOUT) -> LSTMClassifier:

    return LSTMClassifier(sequence_input_size=sequence_input_size,
                          static_input_size=static_input_size,
                          hidden_size=hidden_size,
                          dropout=dropout)



# Warianty A, B i C
def build_model_configs(selected_sec: list[str], model_name: str = 'LSTM') -> list[dict]:
    return [{"name": f"{model_name} A - MARKET SEQUENCE",
            "binary": [],
            "sentiment": []},

            {"name": f"{model_name} B - MARKET SEQUENCE + SEC",
            "binary": selected_sec,
            "sentiment": []},
            
            {"name": f"{model_name} C - MARKET SEQUENCE + SEC + FINBERT",
            "binary": selected_sec + [SENTIMENT_HISTORY_FLAG],
            "sentiment": SENTIMENT_FEATURES}]


# Skalowanie sekwencji
def fit_sequence_scaler(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    if x_train.ndim != 3:
        raise ValueError(f"Tensor sekwencji musi mieć 3 wymiary, otrzymano {x_train.shape}")

    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True)

    std = np.where(std < 1e-8, 1.0, std)

    return (mean.astype(np.float32), std.astype(np.float32))


def transform_sequences(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:

    result = ((x - mean) / std).astype(np.float32)

    if not np.isfinite(result).all():
        raise ValueError("Sekwencje po skalowaniu zawierają NaN lub Inf")

    return result


def prepare_sequence_features(x_train: np.ndarray, x_other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    mean, std = fit_sequence_scaler(x_train)

    return (transform_sequences(x_train, mean, std),
            transform_sequences(x_other, mean, std))


# Cechy statyczne
def build_static_preprocessor(binary_features: list[str],
                            sentiment_features: list[str]) -> ColumnTransformer:

    transformers = [("ticker", OneHotEncoder(handle_unknown="ignore",
                                             sparse_output=False),
                     CATEGORICAL_FEATURES)]

    if binary_features:
        transformers.append(("binary", "passthrough", binary_features))

    if sentiment_features:
        sentiment_pipeline = Pipeline(steps = [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler())
        ])

        transformers.append(("sentiment", sentiment_pipeline, sentiment_features))

    return ColumnTransformer(transformers=transformers,
                             remainder="drop",
                             verbose_feature_names_out=False)

def prepare_static_features(train_df: pd.DataFrame,
                            other_df: pd.DataFrame,
                            binary_features: list[str],
                            sentiment_features: list[str]) -> tuple[np.ndarray, np.ndarray, ColumnTransformer]:

    input_features = list(dict.fromkeys(CATEGORICAL_FEATURES + binary_features
                                        + sentiment_features))

    preprocessor = build_static_preprocessor(binary_features=binary_features,
                                             sentiment_features=sentiment_features)

    x_train = preprocessor.fit_transform(train_df[input_features]).astype(np.float32)

    x_other = preprocessor.transform(other_df[input_features]).astype(np.float32)

    if (not np.isfinite(x_train).all() or not np.isfinite(x_other).all()):
        raise ValueError("Cechy zawierają NaN lub Inf")

    return x_train, x_other, preprocessor


# Dataset Pytorch
def create_loader(sequence_x: np.ndarray,
                  static_x: np.ndarray,
                  y: np.ndarray,
                  batch_size: int,
                  shuffle: bool) -> DataLoader:

    dataset = TensorDataset(torch.from_numpy(np.asarray(sequence_x, dtype=np.float32)),
        torch.from_numpy(np.asarray(static_x, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.float32)))

    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=0)


# Trening jednej epoki 
def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer,
                    criterion,
                    device: torch.device) -> float:

    model.train()

    total_loss = 0.0
    total_count = 0

    for sequence_batch, static_batch, y_batch in loader:
        sequence_batch = sequence_batch.to(device)
        static_batch = static_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(sequence_batch, static_batch)

        loss = criterion(logits, y_batch)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += float(loss.item()) * len(y_batch)
        total_count += len(y_batch)

    return total_loss / total_count


# Strata walidacyjna
def calculate_loss(model: nn.Module,
                   sequence_x: np.ndarray,
                   static_x: np.ndarray,
                   y: np.ndarray,
                   device: torch.device) -> float:

    loader = create_loader(sequence_x=sequence_x,
                           static_x=static_x,
                           y=y,
                           batch_size=BATCH_SIZE,
                           shuffle=False)

    criterion = nn.BCEWithLogitsLoss()

    model.eval()

    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for sequence_batch, static_batch, y_batch in loader:
            sequence_batch = sequence_batch.to(device)
            static_batch = static_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(sequence_batch, static_batch)

            loss = criterion(logits, y_batch)

            total_loss += float(loss.item()) * len(y_batch)
            total_count += len(y_batch)

    return total_loss / total_count


# Wybór liczby epok
def select_best_epoch(sequence_train: np.ndarray,
                      static_train: np.ndarray,
                      y_train: np.ndarray,
                      sequence_val: np.ndarray,
                      static_val: np.ndarray,
                      y_val: np.ndarray,
                      device: torch.device,
                      seed: int,
                      hidden_size: int = HIDDEN_SIZE,
                      dropout: float = DROPOUT,
                      learning_rate: float = LEARNING_RATE,
                      weight_decay: float = WEIGHT_DECAY) -> tuple[int, float]:

    set_seed(seed)

    model = build_lstm_model(sequence_input_size=sequence_train.shape[2],
                             static_input_size=static_train.shape[1],
                             hidden_size=hidden_size,
                             dropout=dropout).to(device)

    loader = create_loader(sequence_x=sequence_train,
                           static_x=static_train,
                           y=y_train,
                           batch_size=BATCH_SIZE,
                           shuffle=True)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=learning_rate,
                                  weight_decay=weight_decay)

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


# Trening
def train_fixed_epochs(sequence_x: np.ndarray,
                       static_x: np.ndarray,
                       y: np.ndarray,
                       device: torch.device,
                       seed: int,
                       epochs: int,
                       hidden_size: int = HIDDEN_SIZE,
                       dropout: float = DROPOUT,
                       learning_rate: float = LEARNING_RATE,
                       weight_decay: float = WEIGHT_DECAY) -> tuple[LSTMClassifier, float]:

    set_seed(seed)

    model = build_lstm_model(sequence_input_size=sequence_x.shape[2],
                             static_input_size=static_x.shape[1],
                             hidden_size=hidden_size,
                             dropout=dropout).to(device)

    loader = create_loader(sequence_x=sequence_x,
                           static_x=static_x,
                           y=y,
                           batch_size=BATCH_SIZE,
                           shuffle=True)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=learning_rate,
                                  weight_decay=weight_decay)

    final_loss = np.nan

    for _ in range(epochs):
        final_loss = train_one_epoch(model=model,
                                     loader=loader,
                                     optimizer=optimizer,
                                     criterion=criterion,
                                     device=device)

    return model, float(final_loss)


# Predykcja
def predict_model(model: nn.Module,
                  sequence_x: np.ndarray,
                  static_x: np.ndarray,
                  device: torch.device) -> np.ndarray:

    loader = create_loader(sequence_x=sequence_x,
                           static_x=static_x,
                           y=np.zeros(len(sequence_x), dtype=np.float32),
                           batch_size=BATCH_SIZE,
                           shuffle=False)

    model.eval()
    probabilities = []

    with torch.no_grad():
        for sequence_batch, static_batch, _ in loader:
            sequence_batch = sequence_batch.to(device)
            static_batch = static_batch.to(device)

            logits = model(sequence_batch, static_batch)

            probabilities.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(probabilities)


# Wczytanie dataseu
def load_sequence_dataset(sequence_file, index_file) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:

    with np.load(sequence_file, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        row_id = data["row_id"].astype(np.int64)
        feature_names = data["feature_names"].astype(str).tolist()

    index_df = pd.read_csv(index_file)

    index_df["Event_Session"] = pd.to_datetime(index_df["Event_Session"], errors="raise")

    if len(X) != len(index_df):
        raise ValueError("Dataset sekwencyjny i indeks mają różną liczbę obserwacji")

    if not np.array_equal(row_id, index_df["Sequence_Row_ID"].to_numpy(dtype=np.int64)):
        raise ValueError("Sequence_Row_ID nie zgadza się między NPZ i indeksem")

    if not np.array_equal(y, index_df[TARGET].astype(int).to_numpy()):
        raise ValueError("Target w NPZ nie zgadza się z indeksem")

    if not np.isfinite(X).all():
        raise ValueError("Dataset sekwencyjny zawiera NaN lub Inf")

    return X, y, index_df, feature_names


# Metadane i cechy statyczne
def prepare_metadata(index_df: pd.DataFrame, model_df: pd.DataFrame) -> pd.DataFrame:

    model_df = prepare_model_dataset(model_df, TARGET)

    validate_target(model_df, TARGET)

    merge_keys = ["Ticker", "Accession", "Event_Session"]

    static_columns = list(dict.fromkeys(merge_keys + ["Abnormal_Event_Return_1D"]
                                        + SEC_BINARY_CANDIDATES + SENTIMENT_FEATURES
                                        + [SENTIMENT_HISTORY_FLAG]))

    missing = [column for column in static_columns if column not in model_df.columns]

    if missing:
        raise ValueError(f"Brakuje kolumn potrzebnych do LSTM: {missing}")

    metadata = index_df.merge(model_df[static_columns],
                              on=merge_keys,
                              how="left",
                              validate="one_to_one")

    metadata = metadata.sort_values("Sequence_Row_ID").reset_index(drop=True)
    

    if not np.array_equal(metadata["Sequence_Row_ID"].to_numpy(),
                          np.arange(len(metadata))):
        raise ValueError("Niepoprawna kolejność Sequence_Row_ID")

    return metadata


def main() -> None:
    for file in [SEQUENCE_FILE, INDEX_FILE, MODEL_DATASET_FILE]:
        if not file.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku: {file}")

    set_seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("LSTM działa na: %s", "GPU" if device.type == "cuda" else "CPU")

    X, y, index_df, feature_names = load_sequence_dataset(SEQUENCE_FILE, INDEX_FILE)

    model_df = pd.read_csv(MODEL_DATASET_FILE)

    metadata = prepare_metadata(index_df=index_df, model_df=model_df)

    if len(metadata) != len(X):
        raise ValueError("Metadata i tensor sekwencji mają różną liczbę obserwacji")

    if not np.array_equal(metadata[TARGET].astype(int).to_numpy(), y):
        raise ValueError("Target metadata nie zgadza się z targetem sekwencji")

    log_repeated_events(metadata)

    logger.info("Target: %s", TARGET)
    logger.info("Liczba obserwacji: %d", len(metadata))
    logger.info("Shape sekwencji: %s", X.shape)
    logger.info("Liczba cech sekwencyjnych: %d", len(feature_names))
    logger.info("Cechy sekwencyjne: %s", feature_names)
    logger.info("Zakres danych: %s -> %s",
                 metadata["Event_Session"].min().date(),
                 metadata["Event_Session"].max().date())
    logger.info("Rozkład targetu:\n%s",
                 metadata[TARGET].value_counts().sort_index().to_string())

    results = []
    all_predictions = []

    years = metadata["Event_Session"].dt.year.to_numpy()

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

        # Skalowanie sekwencji tylko na danych dostępnych w danym etapie
        x_subtrain, x_val = prepare_sequence_features(x_subtrain_raw, x_val_raw)

        x_train, x_test = prepare_sequence_features(x_train_raw, x_test_raw)

        inner_sec = select_sec_features(subtrain_df,
                                        candidates=SEC_BINARY_CANDIDATES,
                                        min_count=MIN_SEC_COUNT)

        final_sec = select_sec_features(train_df,
                                        candidates=SEC_BINARY_CANDIDATES,
                                        min_count=MIN_SEC_COUNT)

        inner_configs = build_model_configs(inner_sec)

        final_configs = build_model_configs(final_sec)

        for variant_id, (inner_config, final_config) in enumerate(zip(inner_configs, final_configs)):
            model_name = final_config["name"]

            seed = (RANDOM_STATE + test_year + variant_id)

            # Cechy statyczne dla wyboru liczby epok
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
            

            # Finalne cechy statyczne - fit wyłącznie na outer TRAIN
            static_train, static_test, preprocessor = prepare_static_features(
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

            # Wyniki na TRAIN - diagnostyka generalizacji
            train_prob = predict_model(model=model,
                                       sequence_x=x_train,
                                       static_x=static_train,
                                       device=device)

            train_pred = (train_prob >= 0.5).astype(int)

            train_metrics = calculate_metrics(y_true=y_train,
                                              y_pred=train_pred,
                                              y_prob=train_prob)
            

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

                      "BA_Gap": (train_metrics["Balanced_Accuracy"]
                                 - test_metrics["Balanced_Accuracy"]),
                      "AUC_Gap": (train_metrics["ROC_AUC"] - test_metrics["ROC_AUC"]),

                      "Selected_SEC_Features": "|".join(feature for feature in final_config["binary"]
                                                        if feature in SEC_BINARY_CANDIDATES),

                      **test_metrics}

            add_confusion_metrics(result=result, y_true=y_test, y_pred=y_pred)

            predictions = test_df[["Ticker", "Event_Session", "Accession", "Abnormal_Event_Return_1D"]].copy()

            predictions["Test_Year"] = test_year
            predictions["Model"] = model_name
            predictions["y_true"] = y_test
            predictions["y_pred"] = y_pred
            predictions["y_prob"] = y_prob

            results.append(result)
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

            del model, preprocessor

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not results:
        raise RuntimeError('Nie udało się wytrenować modeli LSTM')

    results_df = pd.DataFrame(results)

    predictions_df = pd.concat(all_predictions, ignore_index=True)

    summary_df = create_summary(results_df=results_df, predictions_df=predictions_df)

    results_df.to_csv(FOLD_METRICS_FILE, index=False)

    predictions_df.to_csv(PREDICTIONS_FILE, index=False)

    summary_df.to_csv(SUMMARY_FILE, index=False)

    logger.info("Średnie wyniki walk-forward:\n%s", summary_df.to_string(index=False))

    logger.info("Metryki foldów: %s", FOLD_METRICS_FILE)

    logger.info("Predykcje OOS: %s", PREDICTIONS_FILE)

    logger.info("Podsumowanie: %s", SUMMARY_FILE)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    main()
