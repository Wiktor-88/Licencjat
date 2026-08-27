####################################################################################
# Ten plik tworzy sekwencyjna ramke danych dla sieci neuronowych i transformera
####################################################################################

import logging
from pathlib import Path

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# ============================================================
# USTAWIENIA
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = PROJECT_ROOT / "data" / "processed" / "model_dataset.csv"
MARKET_FILE = PROJECT_ROOT / "data" / "processed" / "market_features.csv"

OUTPUT_NPZ = PROJECT_ROOT / "data" / "processed" / "sequence_dataset.npz"
OUTPUT_INDEX = PROJECT_ROOT / "data" / "processed" / "sequence_dataset_index.csv"

BENCHMARK_TICKER = "QQQ"

SEQ_LEN = 30
TARGET = "Target_Abnormal_1D"


# ============================================================
# CECHY SEKWENCYJNE
# ============================================================

STOCK_FEATURES = [
    "Log_Return_1D_Z60",
    "Log_Return_5D_Z60",
    "Volatility_14D_Z60",
    "Relative_Volume_20D_Z60",
    "RSI_14_Scaled",
    "Price_to_SMA20_Z60",
    "Intraday_Return_Z60",
    "Daily_Range_Z60",
]

QQQ_FEATURES = [
    "Log_Return_1D",
    "Log_Return_5D",
    "Volatility_14D",
]

QQQ_SEQUENCE_FEATURES = [
    f"QQQ_{feature}"
    for feature in QQQ_FEATURES
]

SEQUENCE_FEATURES = STOCK_FEATURES + QQQ_SEQUENCE_FEATURES


# ============================================================
# WALIDACJA KOLUMN
# ============================================================

def validate_input_data(
    model: pd.DataFrame,
    market: pd.DataFrame,
) -> None:
    required_model_columns = [
        "Ticker",
        "Accession",
        "Filing_Date",
        "Feature_Cutoff_Session",
        "Event_Session",
        "Publication_Period",
        "Use_In_Primary_Model",
        TARGET,
    ]

    required_market_columns = [
        "Ticker",
        "Date",
        "RSI_14",
        *[
            feature
            for feature in STOCK_FEATURES
            if feature != "RSI_14_Scaled"
        ],
        *QQQ_FEATURES,
    ]

    missing_model_columns = [
        column
        for column in required_model_columns
        if column not in model.columns
    ]

    if missing_model_columns:
        raise ValueError(
            "Brakuje wymaganych kolumn w model_dataset.csv: "
            f"{missing_model_columns}"
        )

    missing_market_columns = [
        column
        for column in required_market_columns
        if column not in market.columns
    ]

    if missing_market_columns:
        raise ValueError(
            "Brakuje wymaganych kolumn w market_features.csv: "
            f"{missing_market_columns}"
        )


# ============================================================
# BUDOWA DATASETU SEKWENCYJNEGO
# ============================================================

def build_sequence_dataset(
    model: pd.DataFrame,
    market: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    model = model.copy()
    market = market.copy()

    validate_input_data(
        model=model,
        market=market,
    )

    # --------------------------------------------------------
    # DATY
    # --------------------------------------------------------

    model["Feature_Cutoff_Session"] = pd.to_datetime(
        model["Feature_Cutoff_Session"],
        errors="raise",
    ).dt.normalize()

    model["Event_Session"] = pd.to_datetime(
        model["Event_Session"],
        errors="raise",
    ).dt.normalize()

    market["Date"] = pd.to_datetime(
        market["Date"],
        errors="raise",
    ).dt.normalize()

    # --------------------------------------------------------
    # PRIMARY DATASET
    # --------------------------------------------------------

    events = model.loc[
        (model["Use_In_Primary_Model"] == 1)
        & model[TARGET].notna()
    ].copy()

    if events.empty:
        raise ValueError(
            "Brak obserwacji do budowy datasetu sekwencyjnego."
        )

    events[TARGET] = events[TARGET].astype(int)

    invalid_targets = ~events[TARGET].isin([0, 1])

    if invalid_targets.any():
        invalid_rows = events.loc[
            invalid_targets,
            ["Ticker", "Accession", TARGET],
        ]

        raise ValueError(
            "Target sekwencyjny musi przyjmować wartości 0/1:\n"
            f"{invalid_rows.to_string(index=False)}"
        )

    events = (
        events
        .sort_values(
            [
                "Event_Session",
                "Ticker",
                "Accession",
            ]
        )
        .reset_index(drop=True)
    )

    events["Sequence_Row_ID"] = np.arange(
        len(events),
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # SKALOWANIE RSI
    # --------------------------------------------------------

    market["RSI_14_Scaled"] = (
        pd.to_numeric(
            market["RSI_14"],
            errors="coerce",
        )
        - 50.0
    ) / 50.0

    # --------------------------------------------------------
    # QQQ
    # --------------------------------------------------------

    qqq = market.loc[
        market["Ticker"] == BENCHMARK_TICKER,
        ["Date", *QQQ_FEATURES],
    ].copy()

    if qqq.empty:
        raise ValueError(
            f"Nie znaleziono benchmarku {BENCHMARK_TICKER} "
            "w market_features.csv."
        )

    if qqq["Date"].duplicated().any():
        duplicate_dates = qqq.loc[
            qqq["Date"].duplicated(keep=False),
            ["Date"],
        ]

        raise ValueError(
            f"Znaleziono zduplikowane sesje {BENCHMARK_TICKER}:\n"
            f"{duplicate_dates.to_string(index=False)}"
        )

    qqq = qqq.rename(
        columns={
            feature: f"QQQ_{feature}"
            for feature in QQQ_FEATURES
        }
    )

    qqq.sort_values(
        "Date",
        inplace=True,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # SPÓŁKI
    # --------------------------------------------------------

    stocks = market.loc[
        market["Ticker"] != BENCHMARK_TICKER
    ].copy()

    if stocks.empty:
        raise ValueError(
            "Brak danych rynkowych spółek."
        )

    duplicate_stock_mask = stocks.duplicated(
        subset=["Ticker", "Date"],
        keep=False,
    )

    if duplicate_stock_mask.any():
        duplicate_rows = stocks.loc[
            duplicate_stock_mask,
            ["Ticker", "Date"],
        ]

        raise ValueError(
            "Znaleziono zduplikowane obserwacje Ticker + Date:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )

    stocks = stocks.merge(
        qqq,
        on="Date",
        how="left",
        validate="many_to_one",
    )

    stocks.sort_values(
        ["Ticker", "Date"],
        inplace=True,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # SEKWENCJE
    # --------------------------------------------------------

    sequences = []
    targets = []

    for _, event in events.iterrows():
        ticker = event["Ticker"]
        accession = event["Accession"]
        cutoff = event["Feature_Cutoff_Session"]

        history = (
            stocks.loc[
                (stocks["Ticker"] == ticker)
                & (stocks["Date"] <= cutoff)
            ]
            .sort_values("Date")
            .tail(SEQ_LEN)
        )

        if len(history) != SEQ_LEN:
            raise ValueError(
                "Niepełna sekwencja historyczna: "
                f"Ticker={ticker}, "
                f"Accession={accession}, "
                f"length={len(history)}, "
                f"expected={SEQ_LEN}."
            )

        if history["Date"].max() != cutoff:
            raise ValueError(
                "Ostatnia sesja sekwencji nie jest "
                "Feature_Cutoff_Session: "
                f"Ticker={ticker}, "
                f"Accession={accession}, "
                f"cutoff={cutoff.date()}, "
                f"last_session={history['Date'].max().date()}."
            )

        expected_dates = (
            qqq.loc[
                qqq["Date"] <= cutoff,
                "Date",
            ]
            .tail(SEQ_LEN)
            .reset_index(drop=True)
        )

        history_dates = (
            history["Date"]
            .reset_index(drop=True)
        )

        if len(expected_dates) != SEQ_LEN:
            raise ValueError(
                "Benchmark nie ma pełnej historii "
                f"{SEQ_LEN} sesji przed cutoff={cutoff.date()}."
            )

        if not history_dates.equals(expected_dates):
            raise ValueError(
                "Historia spółki nie pokrywa dokładnie "
                f"{SEQ_LEN} kolejnych sesji rynkowych: "
                f"Ticker={ticker}, Accession={accession}."
            )

        x = history[
            SEQUENCE_FEATURES
        ].to_numpy(
            dtype=np.float32,
        )

        if x.shape != (
            SEQ_LEN,
            len(SEQUENCE_FEATURES),
        ):
            raise ValueError(
                "Niepoprawny shape pojedynczej sekwencji: "
                f"Ticker={ticker}, "
                f"Accession={accession}, "
                f"shape={x.shape}."
            )

        if not np.isfinite(x).all():
            raise ValueError(
                "NaN lub Inf w sekwencji: "
                f"Ticker={ticker}, "
                f"Accession={accession}."
            )

        sequences.append(x)
        targets.append(
            int(event[TARGET])
        )

    # --------------------------------------------------------
    # FINALNY TENSOR
    # --------------------------------------------------------

    X = np.stack(
        sequences,
    ).astype(
        np.float32,
    )

    y = np.asarray(
        targets,
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # WALIDACJE KOŃCOWE
    # --------------------------------------------------------

    expected_shape = (
        len(events),
        SEQ_LEN,
        len(SEQUENCE_FEATURES),
    )

    if X.shape != expected_shape:
        raise ValueError(
            "Niepoprawny shape X: "
            f"{X.shape}. "
            f"Oczekiwano: {expected_shape}."
        )

    if y.shape != (len(events),):
        raise ValueError(
            "Niepoprawny shape y: "
            f"{y.shape}. "
            f"Oczekiwano: {(len(events),)}."
        )

    expected_targets = events[TARGET].to_numpy(
        dtype=np.int64,
    )

    if not np.array_equal(
        y,
        expected_targets,
    ):
        raise ValueError(
            "Target y nie zgadza się "
            "z kolejnością eventów."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "Finalny tensor X zawiera NaN lub Inf."
        )

    # --------------------------------------------------------
    # INDEX / METADATA
    # --------------------------------------------------------

    index_columns = [
        "Sequence_Row_ID",
        "Ticker",
        "Accession",
        "Filing_Date",
        "Feature_Cutoff_Session",
        "Event_Session",
        "Publication_Period",
        TARGET,
    ]

    index_df = events[
        index_columns
    ].copy()

    index_df["Test_Year"] = (
        index_df["Event_Session"].dt.year
    )

    return X, y, index_df


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku: {MODEL_FILE}"
        )

    if not MARKET_FILE.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku: {MARKET_FILE}"
        )

    model = pd.read_csv(
        MODEL_FILE,
    )

    market = pd.read_csv(
        MARKET_FILE,
    )

    logger.info(
        "Wczytano %d eventów modelowych i %d "
        "obserwacji rynkowych.",
        len(model),
        len(market),
    )

    X, y, index_df = build_sequence_dataset(
        model=model,
        market=market,
    )

    OUTPUT_NPZ.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT_NPZ,
        X=X,
        y=y,
        row_id=index_df[
            "Sequence_Row_ID"
        ].to_numpy(
            dtype=np.int64,
        ),
        feature_names=np.asarray(
            SEQUENCE_FEATURES,
        ),
    )

    index_df.to_csv(
        OUTPUT_INDEX,
        index=False,
    )

    logger.info(
        "Dataset sekwencyjny: X=%s, y=%s, "
        "liczba cech=%d.",
        X.shape,
        y.shape,
        len(SEQUENCE_FEATURES),
    )

    logger.info(
        "Rozkład %s:\n%s",
        TARGET,
        pd.Series(y)
        .value_counts()
        .sort_index()
        .to_string(),
    )

    logger.info(
        "Eventy per rok:\n%s",
        index_df["Test_Year"]
        .value_counts()
        .sort_index()
        .to_string(),
    )

    logger.info(
        "Zapisano dataset sekwencyjny do %s",
        OUTPUT_NPZ,
    )

    logger.info(
        "Zapisano indeks sekwencji do %s",
        OUTPUT_INDEX,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - %(levelname)s - "
            "%(name)s - %(message)s"
        ),
    )

    main()