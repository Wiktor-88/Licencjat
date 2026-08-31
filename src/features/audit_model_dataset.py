# Plik dodatkowy - EDA dla wygenerowanej wcześniej ramki danych

from pathlib import Path
import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "model_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "eda"

TARGET = "Target_Abnormal_1D"


# Sprawdzenie poprawności
def validate_dataset(df: pd.DataFrame) -> None:
    required = [
        "Ticker",
        "Filing_Date",
        "Accession", 
        "Publication_Period",
        "Sentiment_Source", 
        "Use_In_Primary_Model", 
        TARGET
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Brakuje wymaganych kolumn: {missing}")

    duplicates = df.duplicated(["Ticker", "Accession"], keep=False)
    if duplicates.any():
        rows = df.loc[duplicates, ["Ticker", "Accession"]]
        raise ValueError(f"Zduplikowane filingi:\n{rows.to_string(index=False)}")


# Rozkład targetu
def target_summary(df: pd.DataFrame, group: str) -> pd.DataFrame:
    table = pd.crosstab(df[group], df[TARGET])
    table = table.rename(columns={0: "Target_0", 1: "Target_1"})

    for col in ["Target_0", "Target_1"]:
        if col not in table.columns:
            table[col] = 0

    table["Total"] = table["Target_0"] + table["Target_1"]
    table["Target_1_Ratio"] = table["Target_1"] / table["Total"]

    return table.reset_index()

# Zapisywanie wykresów
def save_plot(series: pd.Series, title: str, ylabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)

# Główna funkcja EDA
def run_eda(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["Filing_Date"] = pd.to_datetime(df["Filing_Date"], errors="raise")
    df["Year"] = df["Filing_Date"].dt.year

    primary = df[df["Use_In_Primary_Model"] == 1].copy()

    if primary.empty:
        raise ValueError("Brak obserwacji primary model")

    if primary[TARGET].isna().any():
        raise ValueError("Primary dataset zawiera brakujące wartości targetu")

    primary[TARGET] = primary[TARGET].astype(int)

    # Podstawowe informacje
    summary = {
        "rows_all": len(df),
        "rows_primary": len(primary),
        "rows_excluded": len(df) - len(primary),
        "tickers": int(df["Ticker"].nunique()),
        "duplicates": int(df.duplicated(["Ticker", "Accession"]).sum()),
        "target_0": int((primary[TARGET] == 0).sum()),
        "target_1": int((primary[TARGET] == 1).sum()),
        "target_1_ratio": float((primary[TARGET] == 1).mean()),
        "missing_values_primary": int(primary.isna().sum().sum()),
    }

    numeric = primary.select_dtypes(include=np.number)
    summary["infinite_numeric_values"] = int(np.isinf(numeric).sum().sum())

    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=4, ensure_ascii=False), encoding="utf-8")

    # Braki danych
    missing = primary.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0].rename("Missing_Count").to_frame()
    missing["Missing_Ratio"] = missing["Missing_Count"] / len(primary)
    missing.to_csv(OUTPUT_DIR / "missing_values.csv")

    # Target
    target_counts = primary[TARGET].value_counts().sort_index()
    target_counts.rename("Count").to_csv(OUTPUT_DIR / "target_distribution.csv")

    by_ticker = target_summary(primary, "Ticker")
    by_ticker.to_csv(OUTPUT_DIR / "target_by_ticker.csv", index=False)

    by_year = target_summary(primary, "Year")
    by_year.to_csv(OUTPUT_DIR / "target_by_year.csv", index=False)

    # Źródło sentymentu
    sentiment_source = pd.crosstab(primary["Sentiment_Source"], primary[TARGET])
    sentiment_source.to_csv(OUTPUT_DIR / "target_by_sentiment_source.csv")

    # Moment publikacji
    publication = pd.crosstab(primary["Publication_Period"], primary[TARGET])
    publication.to_csv(OUTPUT_DIR / "target_by_publication_period.csv")

    # Statystyki cech numerycznych
    numeric.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T.to_csv(
        OUTPUT_DIR / "numeric_summary.csv")

    # Korelacja z targetem - bez rzeczy użytych do robienia targetu
    leakage_columns = {
        "Target_Event_1D",
        "Target_Abnormal_1D",
        "Event_Return_1D", 
        "QQQ_Event_Return_1D",
        "Abnormal_Event_Return_1D", 
        "Use_In_Primary_Model",
        "Cutoff_Adj_Close", 
        "Event_Adj_Close",
        "QQQ_Cutoff_Adj_Close", 
        "QQQ_Event_Adj_Close",
    }

    feature_columns = [col for col in numeric.columns if col not in leakage_columns and col != TARGET]

    correlations = (
        primary[feature_columns + [TARGET]]
        .corr(numeric_only=True)[TARGET]
        .drop(TARGET)
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .rename("Correlation_With_Target")
    )

    correlations.to_csv(OUTPUT_DIR / "feature_target_correlations.csv")

    # Wykresy
    save_plot(target_counts, "Rozkład Target_Abnormal_1D", "Liczba obserwacji", "target_distribution.png")
    save_plot(primary["Ticker"].value_counts().sort_index(), "Liczba obserwacji per ticker", "Liczba obserwacji", "rows_by_ticker.png")
    save_plot(primary["Year"].value_counts().sort_index(), "Liczba obserwacji per rok", "Liczba obserwacji", "rows_by_year.png")
    save_plot(primary["Sentiment_Source"].value_counts(), "Źródło sentymentu", "Liczba obserwacji", "sentiment_source.png")

    logger.info("EDA zakończone")
    logger.info("Wszystkie obserwacje: %d", len(df))
    logger.info("Primary model: %d", len(primary))
    logger.info("Target 0/1: %d / %d", (primary[TARGET] == 0).sum(), (primary[TARGET] == 1).sum())
    logger.info("Wyniki zapisano w %s", OUTPUT_DIR)


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    validate_dataset(df)
    run_eda(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    main()