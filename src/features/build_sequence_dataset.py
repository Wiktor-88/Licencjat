# Ten plik tworzy sekwencyjną ramke danych dla sieci neuronowych i transformera

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.model_config import MARKET_COMPACT_FEATURES, TARGET
from src.models.model_utils import log_repeated_events, prepare_model_dataset, validate_target


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_FILE = DATA_DIR / "model_dataset.csv"
MARKET_FILE = DATA_DIR / "market_features.csv"

OUTPUT_NPZ = DATA_DIR / "sequence_dataset.npz"
OUTPUT_INDEX = DATA_DIR / "sequence_dataset_index.csv"

BENCHMARK_TICKER = "QQQ"
SEQ_LEN = 30

# Sieci korzystają z osobnego, zwartego zestawu sekwencyjnego.
# RSI jest tylko przeskalowane do zakresu [-1, 1].
STOCK_FEATURES = [ "RSI_14_Scaled" if feature == "RSI_14" else feature
                    for feature in MARKET_COMPACT_FEATURES
                    if not feature.startswith("QQQ_")]

QQQ_SEQUENCE_FEATURES = [feature for feature in MARKET_COMPACT_FEATURES
                        if feature.startswith("QQQ_")]

QQQ_FEATURES = [feature.removeprefix("QQQ_")
                for feature in QQQ_SEQUENCE_FEATURES]

SEQUENCE_FEATURES = STOCK_FEATURES + QQQ_SEQUENCE_FEATURES


def validate_columns(df: pd.DataFrame, required: list[str], file_name: str) -> None:
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f'Brakuje wymaganych kolumn w {file_name}: {missing}')


def prepare_market_data(market: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    market = market.copy()

    required_columns = ["Ticker",
                        "Date",
                        "RSI_14",
                        *[feature for feature in STOCK_FEATURES if feature != "RSI_14_Scaled"],
                        *QQQ_FEATURES]

    validate_columns(market, required_columns, MARKET_FILE.name)

    market["Date"] = pd.to_datetime(market["Date"], errors="raise").dt.normalize()

    market["RSI_14_Scaled"] = (pd.to_numeric(market["RSI_14"], errors="coerce") - 50.0) / 50.0

    
    qqq = market.loc[market["Ticker"] == BENCHMARK_TICKER, ["Date", *QQQ_FEATURES]].copy()

    if qqq.empty:
        raise ValueError(f"Nie znaleziono benchmarku {BENCHMARK_TICKER}")

    if qqq["Date"].duplicated().any():
        duplicates = qqq.loc[qqq["Date"].duplicated(keep=False), ["Date"]]

        raise ValueError(f"Zduplikowane sesje {BENCHMARK_TICKER}:\n" + duplicates.to_string(index=False))

    qqq = qqq.rename(columns={  feature: f"QQQ_{feature}" for feature in QQQ_FEATURES}).sort_values("Date").reset_index(drop=True)

    # Dane poszczególnych spółek
    stocks = market.loc[market["Ticker"] != BENCHMARK_TICKER].copy()

    if stocks.empty:
        raise ValueError("Brak danych rynkowych spółek")

    duplicate_mask = stocks.duplicated(["Ticker", "Date"], keep=False)

    if duplicate_mask.any():
        duplicates = stocks.loc[duplicate_mask, ["Ticker", "Date"]]

        raise ValueError("Zduplikowane obserwacje Ticker + Date:\n"
                        + duplicates.to_string(index=False))

    stocks = (stocks.merge(qqq, on="Date", how="left", validate="many_to_one")
        .sort_values(["Ticker", "Date"]).reset_index(drop=True))

    return stocks, qqq



def build_sequence_dataset(model: pd.DataFrame, market: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:

    required_model_columns = ["Ticker",
                              "Accession",
                              "Filing_Date",
                              "Feature_Cutoff_Session",
                              "Event_Session",
                              "Publication_Period",
                              "Use_In_Primary_Model",
                              "Sentiment_History_Count_3",
                              "Abnormal_Event_Return_1D",
                              "Tradable_Abnormal_Return_1D",
                              TARGET]

    validate_columns(model, required_model_columns, MODEL_FILE.name)

    events = prepare_model_dataset(model, TARGET)

    validate_target(events, TARGET)

    log_repeated_events(events)

    events["Feature_Cutoff_Session"] = pd.to_datetime(events["Feature_Cutoff_Session"],
                                                      errors="raise").dt.normalize()

    events["Event_Session"] = events["Event_Session"].dt.normalize()

    events["Sequence_Row_ID"] = np.arange(len(events), dtype=np.int64)

    stocks, qqq = prepare_market_data(market)

    # Grupujemy
    stocks_by_ticker = {ticker: group.reset_index(drop=True) for ticker, group in stocks.groupby("Ticker", sort=False)}

    sequences = []
    targets = []

    for event in events.itertuples(index=False):
        ticker = event.Ticker
        accession = event.Accession
        cutoff = event.Feature_Cutoff_Session

        ticker_data = stocks_by_ticker.get(ticker)

        if ticker_data is None:
            raise ValueError(f"Brak danych rynkowych dla Ticker={ticker}")

        history = ticker_data.loc[ticker_data["Date"] <= cutoff].tail(SEQ_LEN).reset_index(drop=True)
        

        if len(history) != SEQ_LEN:
            raise ValueError("Niepełna sekwencja historyczna: "
                            f"Ticker={ticker}, Accession={accession}, "
                            f"length={len(history)}, expected={SEQ_LEN}")

        if history["Date"].iloc[-1] != cutoff:
            raise ValueError("Ostatnia sesja sekwencji nie jest Feature_Cutoff_Session: "
                            f"Ticker={ticker}, Accession={accession}, "
                            f"cutoff={cutoff.date()}, "
                            f"last_session={history['Date'].iloc[-1].date()}")

        expected_dates = qqq.loc[qqq["Date"] <= cutoff, "Date"].tail(SEQ_LEN).reset_index(drop=True)
        

        if len(expected_dates) != SEQ_LEN:
            raise ValueError(f"QQQ nie ma pełnej historii {SEQ_LEN} sesji "
                             f"przed cutoff={cutoff.date()}")

        if not history["Date"].equals(expected_dates):
            raise ValueError("Historia spółki nie pokrywa kolejnych sesji QQQ: "
                            f"Ticker={ticker}, Accession={accession}")

        x = history[SEQUENCE_FEATURES].to_numpy(dtype=np.float32)

        if x.shape != (SEQ_LEN, len(SEQUENCE_FEATURES)):
            raise ValueError(f"Niepoprawny rozmiar sekwencji dla {ticker}: {x.shape}")

        if not np.isfinite(x).all():
            raise ValueError("NaN lub Inf w sekwencji: "
                            f"Ticker={ticker}, Accession={accession}")

        sequences.append(x)
        targets.append(int(getattr(event, TARGET)))

    X = np.stack(sequences).astype(np.float32)
    y = np.asarray(targets, dtype=np.int64)

    expected_shape = (len(events), SEQ_LEN, len(SEQUENCE_FEATURES))

    if X.shape != expected_shape:
        raise ValueError(f"Niepoprawny wymiar X: {X.shape}, powinno być {expected_shape}")

    expected_targets = events[TARGET].to_numpy(dtype=np.int64)

    if not np.array_equal(y, expected_targets):
        raise ValueError("Target y nie zgadza się z kolejnością eventów")

    index_columns = ["Sequence_Row_ID",
                     "Ticker",
                     "Accession",
                     "Filing_Date",
                     "Feature_Cutoff_Session",
                     "Event_Session",
                     "Publication_Period",
                     TARGET]

    index_df = events[index_columns].copy()
    index_df["Test_Year"] = index_df["Event_Session"].dt.year

    return X, y, index_df



def main() -> None:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {MODEL_FILE}")

    if not MARKET_FILE.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {MARKET_FILE}")

    model = pd.read_csv(MODEL_FILE)
    market = pd.read_csv(MARKET_FILE)

    logger.info("Wczytano %d eventów modelowych i %d obserwacji rynkowych",
                len(model), len(market),)

    X, y, index_df = build_sequence_dataset(model=model, market=market)

    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(OUTPUT_NPZ, X=X, y=y,
                        row_id=index_df["Sequence_Row_ID"].to_numpy(dtype=np.int64),
                        feature_names=np.asarray(SEQUENCE_FEATURES))

    index_df.to_csv(OUTPUT_INDEX, index=False)

    logger.info("Dataset sekwencyjny: X=%s, y=%s, liczba cech=%d",
                X.shape, y.shape, len(SEQUENCE_FEATURES))

    logger.info("Rozkład %s:\n%s", TARGET, pd.Series(y).value_counts().sort_index().to_string())

    logger.info("Eventy per rok:\n%s", index_df["Test_Year"].value_counts().sort_index().to_string(),)

    logger.info("Cechy sekwencyjne: %s", SEQUENCE_FEATURES)

    logger.info("Zapisano dataset: %s", OUTPUT_NPZ)

    logger.info("Zapisano indeks: %s", OUTPUT_INDEX)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    main()
