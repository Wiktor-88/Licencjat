from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# USTAWIENIA
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_dataset.csv"
)

MARKET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "market_features.csv"
)

OUTPUT_NPZ = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequence_dataset.npz"
)

OUTPUT_INDEX = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequence_dataset_index.csv"
)

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
    "RSI_14",
    "Price_to_SMA20_Z60",
    "Intraday_Return_Z60",
    "Daily_Range_Z60",
]

QQQ_FEATURES = [
    "Log_Return_1D",
    "Log_Return_5D",
    "Volatility_14D",
]

SEQUENCE_FEATURES = (
    STOCK_FEATURES
    + [
        "QQQ_Log_Return_1D",
        "QQQ_Log_Return_5D",
        "QQQ_Volatility_14D",
    ]
)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 80
    )

    print(
        "BUDOWANIE DATASETU SEKWENCYJNEGO"
    )

    print(
        "=" * 80
    )

    model = pd.read_csv(
        MODEL_FILE
    )

    market = pd.read_csv(
        MARKET_FILE
    )

    model["Event_Session"] = pd.to_datetime(
        model["Event_Session"]
    )

    model["Feature_Cutoff_Session"] = pd.to_datetime(
        model["Feature_Cutoff_Session"]
    )

    market["Date"] = pd.to_datetime(
        market["Date"]
    )

    # ========================================================
    # PRIMARY DATASET
    # ========================================================

    events = model[
        (
            model["Use_In_Primary_Model"]
            == 1
        )
        &
        (
            model[TARGET]
            .notna()
        )
    ].copy()

    events[TARGET] = (
        events[TARGET]
        .astype(int)
    )

    events = events.sort_values(
        [
            "Event_Session",
            "Ticker",
            "Accession",
        ]
    ).reset_index(
        drop=True
    )

    events["Sequence_Row_ID"] = (
        np.arange(
            len(events),
            dtype=np.int64,
        )
    )

    # ========================================================
    # QQQ
    # ========================================================

    qqq = (
        market[
            market["Ticker"]
            == "QQQ"
        ][
            [
                "Date",
                *QQQ_FEATURES,
            ]
        ]
        .copy()
    )

    qqq = qqq.rename(
        columns={
            "Log_Return_1D":
                "QQQ_Log_Return_1D",

            "Log_Return_5D":
                "QQQ_Log_Return_5D",

            "Volatility_14D":
                "QQQ_Volatility_14D",
        }
    )

    # ========================================================
    # SPÓŁKI
    # ========================================================

    stocks = market[
        market["Ticker"]
        != "QQQ"
    ].copy()

    stocks = stocks.merge(
        qqq,
        on="Date",
        how="left",
        validate="many_to_one",
    )

    stocks = stocks.sort_values(
        [
            "Ticker",
            "Date",
        ]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # SEKWENCJE
    # ========================================================

    sequences = []
    targets = []

    for row_id, event in events.iterrows():

        ticker = event["Ticker"]

        cutoff = (
            event[
                "Feature_Cutoff_Session"
            ]
        )

        history = (
            stocks[
                (
                    stocks["Ticker"]
                    == ticker
                )
                &
                (
                    stocks["Date"]
                    <= cutoff
                )
            ]
            .sort_values(
                "Date"
            )
            .tail(
                SEQ_LEN
            )
        )

        if len(history) != SEQ_LEN:
            raise ValueError(
                f"Niepełna sekwencja: "
                f"{ticker}, "
                f"{event['Accession']}, "
                f"length={len(history)}"
            )

        if (
            history["Date"].max()
            != cutoff
        ):
            raise ValueError(
                "Ostatnia sesja nie jest "
                "Feature_Cutoff_Session:\n"
                f"{ticker} "
                f"{event['Accession']}"
            )

        x = (
            history[
                SEQUENCE_FEATURES
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        if not np.isfinite(
            x
        ).all():
            raise ValueError(
                "NaN/Inf w sekwencji:\n"
                f"{ticker} "
                f"{event['Accession']}"
            )

        sequences.append(
            x
        )

        targets.append(
            int(
                event[TARGET]
            )
        )

    X = np.stack(
        sequences
    ).astype(
        np.float32
    )

    y = np.asarray(
        targets,
        dtype=np.int64,
    )

    # ========================================================
    # WALIDACJE KOŃCOWE
    # ========================================================

    expected_shape = (
        len(events),
        SEQ_LEN,
        len(SEQUENCE_FEATURES),
    )

    if X.shape != expected_shape:
        raise ValueError(
            "Niepoprawny shape X:\n"
            f"{X.shape}\n"
            f"Oczekiwano:\n"
            f"{expected_shape}"
        )

    if len(y) != len(events):
        raise ValueError(
            "X i y mają różne długości."
        )

    if not np.array_equal(
        y,
        events[TARGET]
        .to_numpy(
            dtype=np.int64
        ),
    ):
        raise ValueError(
            "Target nie zgadza się "
            "z kolejnością eventów."
        )

    # ========================================================
    # ZAPIS NPZ
    # ========================================================

    np.savez_compressed(
        OUTPUT_NPZ,
        X=X,
        y=y,
        row_id=events[
            "Sequence_Row_ID"
        ].to_numpy(
            dtype=np.int64
        ),
        feature_names=np.asarray(
            SEQUENCE_FEATURES
        ),
    )

    # ========================================================
    # INDEX / METADATA
    # ========================================================

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

    index_df[
        "Test_Year"
    ] = (
        index_df[
            "Event_Session"
        ]
        .dt.year
    )

    index_df.to_csv(
        OUTPUT_INDEX,
        index=False,
    )

    # ========================================================
    # PODSUMOWANIE
    # ========================================================

    print(
        "\nShape X:"
    )
    print(
        X.shape
    )

    print(
        "\nShape y:"
    )
    print(
        y.shape
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
        "\nLiczba cech na sesję:"
    )
    print(
        len(SEQUENCE_FEATURES)
    )

    print(
        "\nCechy:"
    )

    for feature in SEQUENCE_FEATURES:
        print(
            feature
        )

    print(
        "\nEventy per rok:"
    )
    print(
        index_df[
            "Test_Year"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nNPZ zapisano do:"
    )
    print(
        OUTPUT_NPZ
    )

    print(
        "\nIndex zapisano do:"
    )
    print(
        OUTPUT_INDEX
    )