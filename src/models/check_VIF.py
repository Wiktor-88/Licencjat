from pathlib import Path

import pandas as pd

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_dataset.csv"
)


MARKET_FEATURES = [
    "Stock_vs_QQQ_1D",
    "Stock_vs_QQQ_3D",
    "Stock_vs_QQQ_5D",
    "Volatility_14D",
    "Relative_Volume_20D",
    "RSI_14",
    "Price_to_SMA20",
    "Intraday_Return",
    "Daily_Range",
    "QQQ_Log_Return_1D",
    "QQQ_Log_Return_3D",
    "QQQ_Log_Return_5D",
    "QQQ_Volatility_14D",
]


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


def calculate_vif(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:

    X = df[features].copy()

    # VIF powinien być liczony z wyrazem wolnym.
    X_with_const = add_constant(
        X,
        has_constant="add",
    )

    rows = []

    for i, feature in enumerate(
        X_with_const.columns
    ):

        if feature == "const":
            continue

        vif = variance_inflation_factor(
            X_with_const.values,
            i,
        )

        rows.append(
            {
                "Feature": feature,
                "VIF": vif,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "VIF",
            ascending=False,
        )
    )


def main():

    df = pd.read_csv(
        DATA_PATH
    )

    df["Event_Session"] = pd.to_datetime(
        df["Event_Session"]
    )

    df = df[
        (df["Use_In_Primary_Model"] == 1)
        & df["Target_Abnormal_1D"].notna()
    ].copy()

    # --------------------------------------------------------
    # Bierzemy tylko dane historyczne do końca 2024.
    # Nie używamy 2025/2026 do diagnostyki cech.
    # --------------------------------------------------------

    train_df = df[
        df["Event_Session"].dt.year <= 2024
    ].copy()

    print(
        "Liczba obserwacji TRAIN:",
        len(train_df),
    )

    print(
        "\n"
        + "=" * 70
    )
    print(
        "VIF - RAW MARKET"
    )
    print(
        "=" * 70
    )

    print(
        calculate_vif(
            train_df,
            MARKET_FEATURES,
        ).to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 70
    )
    print(
        "VIF - MARKET Z60"
    )
    print(
        "=" * 70
    )

    print(
        calculate_vif(
            train_df,
            MARKET_Z_FEATURES,
        ).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()