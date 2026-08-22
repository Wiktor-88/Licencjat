from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

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

SEQUENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequence_dataset.npz"
)

INDEX_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequence_dataset_index.csv"
)

MODEL_DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_dataset.csv"
)

OUTPUT_PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lstm_oos_predictions.csv"
)

OUTPUT_RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lstm_walk_forward_results.csv"
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

MAX_EPOCHS = 80
PATIENCE = 8
MIN_DELTA = 1e-4

BATCH_SIZE = 64

HIDDEN_SIZE = 32

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

DROPOUT = 0.20


# ============================================================
# FINBERT
# ============================================================

SENTIMENT_FEATURES = [
    "Mean_Net_Sentiment",
    "Sentiment_Momentum_3",
]


# ============================================================
# REPRODUKOWALNOŚĆ
# ============================================================

def set_seed(
    seed: int,
) -> None:

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# MODEL
# ============================================================

class LSTMClassifier(
    nn.Module
):

    def __init__(
        self,
        sequence_input_size: int,
        static_input_size: int,
        hidden_size: int = HIDDEN_SIZE,
        dropout: float = DROPOUT,
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=sequence_input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.sequence_dropout = nn.Dropout(
            dropout
        )

        combined_size = (
            hidden_size
            + static_input_size
        )

        self.head = nn.Sequential(

            nn.Linear(
                combined_size,
                32,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                32,
                1,
            ),
        )

    def forward(
        self,
        sequence_x,
        static_x,
    ):

        _, (
            hidden_state,
            _
        ) = self.lstm(
            sequence_x
        )

        # Ostatni hidden state.
        sequence_representation = (
            hidden_state[-1]
        )

        sequence_representation = (
            self.sequence_dropout(
                sequence_representation
            )
        )

        combined = torch.cat(
            [
                sequence_representation,
                static_x,
            ],
            dim=1,
        )

        logits = self.head(
            combined
        )

        return logits.squeeze(
            dim=1
        )


# ============================================================
# SEC FEATURE SELECTION
# ============================================================

def select_sec_features(
    train_df: pd.DataFrame,
) -> list[str]:

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


# ============================================================
# SKALOWANIE SEKWENCJI
# ============================================================

def fit_sequence_scaler(
    x_train: np.ndarray,
):

    mean = np.mean(
        x_train,
        axis=(0, 1),
        keepdims=True,
    )

    std = np.std(
        x_train,
        axis=(0, 1),
        keepdims=True,
    )

    std = np.where(
        std < 1e-8,
        1.0,
        std,
    )

    return (
        mean.astype(
            np.float32
        ),
        std.astype(
            np.float32
        ),
    )


def transform_sequences(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:

    result = (
        (x - mean)
        / std
    )

    return result.astype(
        np.float32
    )


# ============================================================
# SENTIMENT SCALER
# ============================================================

def fit_sentiment_scaler(
    train_df: pd.DataFrame,
):

    stats = {}

    for column in SENTIMENT_FEATURES:

        values = pd.to_numeric(
            train_df[column],
            errors="coerce",
        )

        mean = values.mean()

        std = values.std(
            ddof=0
        )

        if pd.isna(mean):
            mean = 0.0

        if (
            pd.isna(std)
            or std < 1e-8
        ):
            std = 1.0

        stats[column] = (
            float(mean),
            float(std),
        )

    return stats


# ============================================================
# STATIC FEATURES
# ============================================================

def build_static_features(
    df: pd.DataFrame,
    ticker_categories: list[str],
    sec_features: list[str],
    include_sec: bool,
    include_sentiment: bool,
    sentiment_stats=None,
):

    arrays = []
    feature_names = []

    # --------------------------------------------------------
    # TICKER ONE-HOT
    # --------------------------------------------------------

    ticker_values = (
        df["Ticker"]
        .astype(str)
    )

    known_tickers = set(
        ticker_categories
    )

    unknown_tickers = (
        set(
            ticker_values.unique()
        )
        - known_tickers
    )

    if unknown_tickers:
        raise ValueError(
            "Ticker w TEST niewidziany w TRAIN:\n"
            + "\n".join(
                sorted(
                    unknown_tickers
                )
            )
        )

    ticker_matrix = np.column_stack(
        [
            (
                ticker_values
                == ticker
            )
            .astype(
                np.float32
            )
            .to_numpy()
            for ticker in ticker_categories
        ]
    )

    arrays.append(
        ticker_matrix
    )

    feature_names.extend(
        [
            f"Ticker_{ticker}"
            for ticker in ticker_categories
        ]
    )

    # --------------------------------------------------------
    # SEC
    # --------------------------------------------------------

    if include_sec:

        sec_matrix = (
            df[
                sec_features
            ]
            .fillna(0)
            .astype(
                np.float32
            )
            .to_numpy()
        )

        arrays.append(
            sec_matrix
        )

        feature_names.extend(
            sec_features
        )

    # --------------------------------------------------------
    # FINBERT
    # --------------------------------------------------------

    if include_sentiment:

        if sentiment_stats is None:
            raise ValueError(
                "Brak sentiment_stats."
            )

        sentiment_arrays = []

        for column in SENTIMENT_FEATURES:

            mean, std = (
                sentiment_stats[column]
            )

            values = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            standardized = (
                (values - mean)
                / std
            )

            # Brak sentimentu:
            # 0 po standaryzacji = średnia TRAIN.
            standardized = (
                standardized
                .fillna(0.0)
            )

            sentiment_arrays.append(
                standardized
                .astype(
                    np.float32
                )
                .to_numpy()
            )

        sentiment_matrix = np.column_stack(
            sentiment_arrays
        )

        arrays.append(
            sentiment_matrix
        )

        feature_names.extend(
            SENTIMENT_FEATURES
        )

    result = np.concatenate(
        arrays,
        axis=1,
    ).astype(
        np.float32
    )

    return (
        result,
        feature_names,
    )


# ============================================================
# DATASET PYTORCH
# ============================================================

def create_loader(
    sequence_x,
    static_x,
    y,
    batch_size,
    shuffle,
):

    dataset = TensorDataset(

        torch.tensor(
            sequence_x,
            dtype=torch.float32,
        ),

        torch.tensor(
            static_x,
            dtype=torch.float32,
        ),

        torch.tensor(
            y,
            dtype=torch.float32,
        ),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


# ============================================================
# TRAINING, LOSS
# ============================================================
def calculate_loss(
    model,
    sequence_x,
    static_x,
    y,
    device,
):

    loader = create_loader(
        sequence_x=sequence_x,
        static_x=static_x,
        y=y,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    criterion = nn.BCEWithLogitsLoss(
        reduction="sum"
    )

    model.eval()

    total_loss = 0.0
    total_count = 0

    with torch.no_grad():

        for (
            sequence_batch,
            static_batch,
            y_batch,
        ) in loader:

            sequence_batch = (
                sequence_batch.to(device)
            )

            static_batch = (
                static_batch.to(device)
            )

            y_batch = (
                y_batch.to(device)
            )

            logits = model(
                sequence_batch,
                static_batch,
            )

            loss = criterion(
                logits,
                y_batch,
            )

            total_loss += float(
                loss.item()
            )

            total_count += len(
                y_batch
            )

    return (
        total_loss
        / total_count
    )


def select_best_epoch(
    sequence_train,
    static_train,
    y_train,
    sequence_val,
    static_val,
    y_val,
    device,
    seed,
):

    set_seed(
        seed
    )

    model = LSTMClassifier(
        sequence_input_size=(
            sequence_train.shape[2]
        ),
        static_input_size=(
            static_train.shape[1]
        ),
    ).to(device)

    loader = create_loader(
        sequence_x=sequence_train,
        static_x=static_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_epoch = 1
    best_val_loss = np.inf

    epochs_without_improvement = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        model.train()

        train_losses = []

        for (
            sequence_batch,
            static_batch,
            y_batch,
        ) in loader:

            sequence_batch = (
                sequence_batch.to(device)
            )

            static_batch = (
                static_batch.to(device)
            )

            y_batch = (
                y_batch.to(device)
            )

            optimizer.zero_grad()

            logits = model(
                sequence_batch,
                static_batch,
            )

            loss = criterion(
                logits,
                y_batch,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            train_losses.append(
                float(
                    loss.item()
                )
            )

        train_loss = float(
            np.mean(
                train_losses
            )
        )

        val_loss = calculate_loss(
            model=model,
            sequence_x=sequence_val,
            static_x=static_val,
            y=y_val,
            device=device,
        )

        if (
            val_loss
            <
            best_val_loss
            - MIN_DELTA
        ):

            best_val_loss = (
                val_loss
            )

            best_epoch = epoch

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epoch == 1
            or epoch % 10 == 0
        ):

            print(
                f"    Epoch {epoch:02d} "
                f"- train={train_loss:.6f} "
                f"- val={val_loss:.6f}"
            )

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "    Early stopping "
                f"at epoch {epoch}"
            )

            break

    print(
        f"    Best epoch: {best_epoch}"
    )

    print(
        f"    Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        best_epoch,
        best_val_loss,
    )


def train_fixed_epochs(
    sequence_x,
    static_x,
    y,
    device,
    seed,
    epochs,
):

    set_seed(
        seed
    )

    model = LSTMClassifier(
        sequence_input_size=(
            sequence_x.shape[2]
        ),
        static_input_size=(
            static_x.shape[1]
        ),
    ).to(device)

    loader = create_loader(
        sequence_x=sequence_x,
        static_x=static_x,
        y=y,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    final_loss = np.nan

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        epoch_losses = []

        for (
            sequence_batch,
            static_batch,
            y_batch,
        ) in loader:

            sequence_batch = (
                sequence_batch.to(device)
            )

            static_batch = (
                static_batch.to(device)
            )

            y_batch = (
                y_batch.to(device)
            )

            optimizer.zero_grad()

            logits = model(
                sequence_batch,
                static_batch,
            )

            loss = criterion(
                logits,
                y_batch,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            epoch_losses.append(
                float(
                    loss.item()
                )
            )

        final_loss = float(
            np.mean(
                epoch_losses
            )
        )

    return (
        model,
        final_loss,
    )





# ============================================================
# PREDYKCJA
# ============================================================

def predict_model(
    model,
    sequence_x,
    static_x,
    device,
):

    model.eval()

    loader = create_loader(
        sequence_x=sequence_x,
        static_x=static_x,
        y=np.zeros(
            len(sequence_x),
            dtype=np.float32,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    probabilities = []

    with torch.no_grad():

        for (
            sequence_batch,
            static_batch,
            _,
        ) in loader:

            sequence_batch = (
                sequence_batch.to(
                    device
                )
            )

            static_batch = (
                static_batch.to(
                    device
                )
            )

            logits = model(
                sequence_batch,
                static_batch,
            )

            probs = torch.sigmoid(
                logits
            )

            probabilities.append(
                probs.cpu()
                .numpy()
            )

    return np.concatenate(
        probabilities
    )


# ============================================================
# METRYKI
# ============================================================

def calculate_metrics(
    y_true,
    y_prob,
):

    y_pred = (
        y_prob >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    if np.unique(
        y_true
    ).size == 2:

        roc = roc_auc_score(
            y_true,
            y_prob,
        )

    else:
        roc = np.nan

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    return {
        "Accuracy":
            accuracy,

        "Balanced_Accuracy":
            balanced,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1":
            f1,

        "ROC_AUC":
            roc,

        "y_pred":
            y_pred,

        "Confusion_Matrix":
            cm,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    set_seed(
        RANDOM_STATE
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "LSTM WALK-FORWARD"
    )

    print(
        "=" * 80
    )

    print(
        "\nDevice:"
    )

    print(
        device
    )

    # ========================================================
    # SEKWENCJE
    # ========================================================

    sequence_data = np.load(
        SEQUENCE_FILE
    )

    X = (
        sequence_data["X"]
        .astype(
            np.float32
        )
    )

    y = (
        sequence_data["y"]
        .astype(
            np.int64
        )
    )

    feature_names = (
        sequence_data[
            "feature_names"
        ]
        .tolist()
    )

    index_df = pd.read_csv(
        INDEX_FILE
    )

    index_df[
        "Event_Session"
    ] = pd.to_datetime(
        index_df[
            "Event_Session"
        ]
    )

    # ========================================================
    # STATIC FEATURES Z MODEL DATASET
    # ========================================================

    model_df = pd.read_csv(
        MODEL_DATASET_FILE
    )

    model_df[
        "Event_Session"
    ] = pd.to_datetime(
        model_df[
            "Event_Session"
        ]
    )

    model_df = model_df[
        (
            model_df[
                "Use_In_Primary_Model"
            ]
            == 1
        )
        &
        (
            model_df[TARGET]
            .notna()
        )
    ].copy()

    # --------------------------------------------------------
    # Łączymy index sekwencji z pełnymi cechami eventu.
    # --------------------------------------------------------

    merge_keys = [
        "Ticker",
        "Accession",
        "Event_Session",
    ]

    static_columns = [
        *merge_keys,
        "Abnormal_Event_Return_1D",
        "Has_EX99",
        "Mean_Net_Sentiment",
        "Sentiment_Momentum_3",
    ]

    static_columns.extend(
        [
            column
            for column in model_df.columns
            if column.startswith(
                "Has_Item_"
            )
        ]
    )

    static_columns = list(
        dict.fromkeys(
            static_columns
        )
    )

    metadata = index_df.merge(
        model_df[
            static_columns
        ],
        on=merge_keys,
        how="left",
        validate="one_to_one",
    )

    metadata = metadata.sort_values(
        "Sequence_Row_ID"
    ).reset_index(
        drop=True
    )

    # ========================================================
    # WALIDACJE
    # ========================================================

    if len(metadata) != len(X):

        raise ValueError(
            "Metadata i X mają różne "
            "liczby obserwacji."
        )

    if not np.array_equal(
        metadata[TARGET]
        .astype(int)
        .to_numpy(),
        y,
    ):

        raise ValueError(
            "Target metadata != target NPZ."
        )

    if not np.array_equal(
        metadata[
            "Sequence_Row_ID"
        ].to_numpy(),
        np.arange(
            len(metadata)
        ),
    ):

        raise ValueError(
            "Niepoprawna kolejność "
            "Sequence_Row_ID."
        )

    print(
        "\nShape X:"
    )

    print(
        X.shape
    )

    print(
        "\nLiczba obserwacji:"
    )

    print(
        len(metadata)
    )

    print(
        "\nTarget:"
    )

    print(
        pd.Series(y)
        .value_counts()
        .sort_index()
    )

    print(
        "\nCechy sekwencyjne:"
    )

    for feature in feature_names:
        print(
            feature
        )

    results = []
    all_predictions = []

    # ========================================================
    # WALK-FORWARD
    # ========================================================

    for test_year in TEST_YEARS:

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"TEST YEAR: "
            f"{test_year}"
        )

        print(
            "=" * 80
        )

        year_values = (
            metadata[
                "Event_Session"
            ]
            .dt.year
            .to_numpy()
        )

        train_indices = np.where(
            year_values
            < test_year
        )[0]

        test_indices = np.where(
            year_values
            == test_year
        )[0]

        train_df = metadata.iloc[
            train_indices
        ].copy()

        test_df = metadata.iloc[
            test_indices
        ].copy()

        y_train = y[
            train_indices
        ]

        y_test = y[
            test_indices
        ]

        x_train_raw = X[
            train_indices
        ]

        x_test_raw = X[
            test_indices
        ]

        print(
            "\nTrain size:",
            len(train_indices),
        )

        print(
            "Test size:",
            len(test_indices),
        )

        print(
            "\nTarget TRAIN:"
        )

        print(
            pd.Series(
                y_train
            )
            .value_counts()
            .sort_index()
        )

        print(
            "\nTarget TEST:"
        )

        print(
            pd.Series(
                y_test
            )
            .value_counts()
            .sort_index()
        )

        # ====================================================
        # INNER TEMPORAL VALIDATION
        # ====================================================

        validation_year = (
            test_year - 1
        )

        train_years = (
            train_df[
                "Event_Session"
            ]
            .dt.year
            .to_numpy()
        )

        subtrain_positions = np.where(
            train_years
            < validation_year
        )[0]

        val_positions = np.where(
            train_years
            == validation_year
        )[0]

        if (
            len(subtrain_positions) == 0
            or len(val_positions) == 0
        ):
            raise ValueError(
                "Brak danych dla temporal "
                "subtrain/validation."
            )

        subtrain_df = (
            train_df.iloc[
                subtrain_positions
            ]
            .copy()
        )

        val_df = (
            train_df.iloc[
                val_positions
            ]
            .copy()
        )

        y_subtrain = (
            y_train[
                subtrain_positions
            ]
        )

        y_val = (
            y_train[
                val_positions
            ]
        )

        x_subtrain_raw = (
            x_train_raw[
                subtrain_positions
            ]
        )

        x_val_raw = (
            x_train_raw[
                val_positions
            ]
        )

        print(
            "\nTemporal validation:"
        )

        print(
            "Subtrain:",
            len(subtrain_df),
        )

        print(
            f"Validation {validation_year}:",
            len(val_df),
        )

        # ====================================================
        # INNER SCALER - SUBTRAIN ONLY
        # ====================================================

        inner_mean, inner_std = (
            fit_sequence_scaler(
                x_subtrain_raw
            )
        )

        x_subtrain = (
            transform_sequences(
                x_subtrain_raw,
                inner_mean,
                inner_std,
            )
        )

        x_val = (
            transform_sequences(
                x_val_raw,
                inner_mean,
                inner_std,
            )
        )

        inner_tickers = sorted(
            subtrain_df[
                "Ticker"
            ]
            .astype(str)
            .unique()
        )

        inner_sec_features = (
            select_sec_features(
                subtrain_df
            )
        )

        inner_sentiment_stats = (
            fit_sentiment_scaler(
                subtrain_df
            )
        )

        # ====================================================
        # FINAL PREPROCESSING - PEŁNY OUTER TRAIN
        # ====================================================

        sequence_mean, sequence_std = (
            fit_sequence_scaler(
                x_train_raw
            )
        )

        x_train = (
            transform_sequences(
                x_train_raw,
                sequence_mean,
                sequence_std,
            )
        )

        x_test = (
            transform_sequences(
                x_test_raw,
                sequence_mean,
                sequence_std,
            )
        )

        ticker_categories = sorted(
            train_df[
                "Ticker"
            ]
            .astype(str)
            .unique()
        )

        sec_features = (
            select_sec_features(
                train_df
            )
        )

        sentiment_stats = (
            fit_sentiment_scaler(
                train_df
            )
        )

        print(
            "\nSEC features FINAL TRAIN:"
        )

        for feature in sec_features:

            print(
                f"{feature}: "
                f"{int(train_df[feature].fillna(0).sum())}"
            )


        for feature in sec_features:

            print(
                f"{feature}: "
                f"{int(train_df[feature].fillna(0).sum())}"
            )

        # ====================================================
        # DEFINICJE A / B / C
        # ====================================================

        variants = [
            {
                "name":
                    "LSTM A - MARKET SEQUENCE",

                "include_sec":
                    False,

                "include_sentiment":
                    False,
            },

            {
                "name":
                    "LSTM B - MARKET SEQUENCE + SEC",

                "include_sec":
                    True,

                "include_sentiment":
                    False,
            },

            {
                "name":
                    "LSTM C - MARKET SEQUENCE + SEC + FINBERT",

                "include_sec":
                    True,

                "include_sentiment":
                    True,
            },
        ]

        # ====================================================
        # TRENING WARIANTÓW
        # ====================================================

        for variant_id, variant in enumerate(
            variants
        ):

            model_name = (
                variant["name"]
            )

            print(
                "\n"
                + "-" * 80
            )

            print(
                model_name
            )

            static_train, static_names = (
                build_static_features(
                    df=train_df,
                    ticker_categories=(
                        ticker_categories
                    ),
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
                    sentiment_stats=(
                        sentiment_stats
                    ),
                )
            )

            static_test, _ = (
                build_static_features(
                    df=test_df,
                    ticker_categories=(
                        ticker_categories
                    ),
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
                    sentiment_stats=(
                        sentiment_stats
                    ),
                )
            )

            print(
                "\nLiczba cech statycznych:",
                static_train.shape[1],
            )


            # ================================================
            # INNER VALIDATION FEATURES
            # ================================================

            static_subtrain, _ = (
                build_static_features(
                    df=subtrain_df,
                    ticker_categories=(
                        inner_tickers
                    ),
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
                    sentiment_stats=(
                        inner_sentiment_stats
                    ),
                )
            )

            static_val, _ = (
                build_static_features(
                    df=val_df,
                    ticker_categories=(
                        inner_tickers
                    ),
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
                    sentiment_stats=(
                        inner_sentiment_stats
                    ),
                )
            )

            print(
                "\nWybór liczby epok..."
            )

            best_epoch, best_val_loss = (
                select_best_epoch(
                    sequence_train=x_subtrain,
                    static_train=static_subtrain,
                    y_train=y_subtrain,
                    sequence_val=x_val,
                    static_val=static_val,
                    y_val=y_val,
                    device=device,
                    seed=(
                        RANDOM_STATE
                        + test_year
                        + variant_id
                    ),
                )
            )


            # ================================================
            # FINAL TRAIN FEATURES
            # ================================================

            static_train, static_names = (
                build_static_features(
                    df=train_df,
                    ticker_categories=(
                        ticker_categories
                    ),
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
                    sentiment_stats=(
                        sentiment_stats
                    ),
                )
            )

            static_test, _ = (
                build_static_features(
                    df=test_df,
                    ticker_categories=(
                        ticker_categories
                    ),
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
                    sentiment_stats=(
                        sentiment_stats
                    ),
                )
            )

            print(
                "\nLiczba cech statycznych:",
                static_train.shape[1],
            )

            print(
                "Final training epochs:",
                best_epoch,
            )

            model, final_train_loss = (
                train_fixed_epochs(
                    sequence_x=x_train,
                    static_x=static_train,
                    y=y_train,
                    device=device,
                    seed=(
                        RANDOM_STATE
                        + test_year
                        + variant_id
                    ),
                    epochs=best_epoch,
                )
            )




            y_prob = predict_model(
                model=model,
                sequence_x=x_test,
                static_x=static_test,
                device=device,
            )

            metrics = (
                calculate_metrics(
                    y_true=y_test,
                    y_prob=y_prob,
                )
            )

            print(
                "\nWyniki:"
            )

            print(
                f"Accuracy: "
                f"{metrics['Accuracy']:.6f}"
            )

            print(
                f"Balanced Accuracy: "
                f"{metrics['Balanced_Accuracy']:.6f}"
            )

            print(
                f"Precision: "
                f"{metrics['Precision']:.6f}"
            )

            print(
                f"Recall: "
                f"{metrics['Recall']:.6f}"
            )

            print(
                f"F1: "
                f"{metrics['F1']:.6f}"
            )

            print(
                f"ROC-AUC: "
                f"{metrics['ROC_AUC']:.6f}"
            )

            print(
                "\nConfusion Matrix:"
            )

            print(
                metrics[
                    "Confusion_Matrix"
                ]
            )

            result = {
                "Test_Year":
                    test_year,

                "Model":
                    model_name,

                "Train_Size":
                    len(train_indices),

                "Test_Size":
                    len(test_indices),

                "Sequence_Length":
                    X.shape[1],

                "Sequence_Features":
                    X.shape[2],

                "Static_Features":
                    static_train.shape[1],

                "Final_Train_Loss":
                    final_train_loss,

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

                "Validation_Year":
                    validation_year,

                "Best_Epoch":
                    best_epoch,

                "Best_Val_Loss":
                    best_val_loss,
            }

            results.append(
                result
            )

            predictions = (
                test_df[
                    [
                        "Ticker",
                        "Event_Session",
                        "Accession",
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
            ] = model_name

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

            # Zwolnienie pamięci GPU.
            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ========================================================
    # PODSUMOWANIE
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_RESULTS_FILE,
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

    # ========================================================
    # PREDYKCJE OOS
    # ========================================================

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
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
        "\nWyniki zapisano do:"
    )

    print(
        OUTPUT_RESULTS_FILE
    )

    print(
        "\nPredykcje zapisano do:"
    )

    print(
        OUTPUT_PREDICTIONS_FILE
    )