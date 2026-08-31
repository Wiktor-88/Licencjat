# Piąty plik - odpoiwada za połączneie analizy sentymentów z danymi SEC

from pathlib import Path

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "sentiment_blocks.csv"
METADATA_FILE = PROJECT_ROOT / "data" / "processed" / "sec_document_metadata.csv"
OUTPUT_FILE = PROJECT_ROOT/ "data" / "processed" / "sentiment_filing_features.csv"


# Wymagane kolumny z sentiment
REQUIRED_COLUMNS = [
    "Ticker",
    "Date",
    "Accession",
    "Source_Type",
    "Item_Number",
    "Block_ID",
    "Predicted_Label",
    "Prob_Positive",
    "Prob_Negative",
    "Prob_Neutral",
    "Net_Sentiment",
]


# Sprawdzeni czy sa wszytkie wymagane kolumny
def validate_dataframe(df: pd.DataFrame) -> None:

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Brakuje wymaganych kolumn w sentiment_blocks.csv: {missing_columns}")


# Agregacja pojedyńczego filingu
def aggregate_single_filing(filing_df: pd.DataFrame) -> dict:

    # Sprawdzamy poprawności
    if filing_df.empty:
        raise ValueError("Nie można agregować pustego filingu")

    if filing_df["Accession"].nunique() != 1:
        raise ValueError("DataFrame zawiera więcej niż jeden filing")

    ticker = filing_df["Ticker"].iloc[0]
    filing_date = filing_df["Date"].iloc[0]
    accession = filing_df["Accession"].iloc[0]


    # Podział na 8K i EX99
    blocks_8k = filing_df[filing_df["Source_Type"] == "8-K"].copy()
    blocks_ex99 = filing_df[filing_df["Source_Type"].str.startswith("EX-99", na=False)].copy()



    # EX-99 jest głównym źródłem sentymentu, gdy nie ma - bierzemy 8K
    if not blocks_ex99.empty:
        sentiment_blocks = blocks_ex99
        sentiment_source = "EX99"
    elif not blocks_8k.empty:
        sentiment_blocks = blocks_8k
        sentiment_source = "8K"
    else:
        raise ValueError(f"Brak bloków do analizy sentymentu dla filingu {accession}")

    # Itemy z głównego 8K
    item_numbers = sorted({str(item_number) for item_number in blocks_8k["Item_Number"]
                          if pd.notna(item_number)
                          and str(item_number) not in {"N/A", "UNKNOWN"}
        })

    source_types = sorted(filing_df["Source_Type"].dropna().unique())

    # Metadane idące na cechy
    result = {
        "Ticker": ticker,
        "Date": filing_date,
        "Accession": accession,
        "Has_8K": int(not blocks_8k.empty),
        "Has_EX99": int(not blocks_ex99.empty),
        "Sentiment_Source": sentiment_source,
        "Filing_Block_Count": len(filing_df),
        "EightK_Block_Count": len(blocks_8k),
        "EX99_Block_Count": len(blocks_ex99),
        "Item_Numbers": "|".join(item_numbers),
        "Source_Types": "|".join(source_types),
    }


    # Liczebność klas
    labels = sentiment_blocks["Predicted_Label"].str.lower()
    sentiment_block_count = len(sentiment_blocks)

    valid_labels ={"positive", "negative", "neutral",}

    unexpected_labels = set(labels.dropna().unique()) - valid_labels

    if labels.isna().any() or unexpected_labels:
        raise ValueError(f"Niepoprawne etykiety sentymentu dla filingu {accession}.")

    positive_count = int((labels == "positive").sum())
    negative_count = int((labels == "negative").sum())
    neutral_count = int((labels == "neutral").sum())

    # Szybjke sprawdzenie:
    classified_block_count = positive_count + negative_count + neutral_count

    if classified_block_count != sentiment_block_count:
        raise ValueError("Nie wszystkie bloki mają poprawną etykietę sentymentu "
                          f"Accession: {accession}")


    # Sprawdzenie czy nie ma samych Nanów
    probability_columns = ["Prob_Positive", "Prob_Negative", "Prob_Neutral", "Net_Sentiment"]

    if sentiment_blocks[probability_columns].isna().any().any():
        raise ValueError(f"Brakujące wartości sentymentu dla filingu {accession}")

    # Najbardziej pozytywny blok
    most_positive_index = sentiment_blocks["Prob_Positive"].idxmax()

    most_positive_row = sentiment_blocks.loc[most_positive_index]
    

    # Najbardziej negatywny blok
    most_negative_index = sentiment_blocks["Prob_Negative"].idxmax()

    most_negative_row = sentiment_blocks.loc[most_negative_index]
    

    # Dodanie cech sentymentu
    result.update({
        "Positive_Block_Count": positive_count,
        "Negative_Block_Count": negative_count,
        "Neutral_Block_Count": neutral_count,

        "Positive_Ratio": positive_count / sentiment_block_count,
        "Negative_Ratio": negative_count / sentiment_block_count,
        "Neutral_Ratio": neutral_count / sentiment_block_count,

        "Mean_Positive": sentiment_blocks["Prob_Positive"].mean(),
        "Mean_Negative": sentiment_blocks["Prob_Negative"].mean(),
        "Mean_Neutral": sentiment_blocks["Prob_Neutral"].mean(),

        "Mean_Net_Sentiment": sentiment_blocks["Net_Sentiment"].mean(),
        "Median_Net_Sentiment": sentiment_blocks["Net_Sentiment"].median(),
        "Min_Net_Sentiment": sentiment_blocks["Net_Sentiment"].min(),
        "Max_Net_Sentiment": sentiment_blocks["Net_Sentiment"].max(),

        "Max_Positive_Probability": sentiment_blocks["Prob_Positive"].max(),
        "Max_Negative_Probability": sentiment_blocks["Prob_Negative"].max(),

        "Most_Positive_Source_Type": most_positive_row["Source_Type"],
        "Most_Positive_Block_ID": int(most_positive_row["Block_ID"]),
        "Most_Negative_Source_Type": most_negative_row["Source_Type"],
        "Most_Negative_Block_ID": int(most_negative_row["Block_ID"]),
    })

    return result


# Cechy dla ITEM z dokumentów 8K
def create_item_features(df: pd.DataFrame) -> pd.DataFrame:

    key_columns = ["Ticker", "Date", "Accession",]

    item_df = df[df["Source_Type"] == "8-K"].copy()

    item_df["Item_Number"] = item_df["Item_Number"].astype("string")

    # Pominięcie 9.01 bo to informacja techiniczna
    item_df = item_df[item_df["Item_Number"].notna()
                      & ~item_df["Item_Number"].isin(["N/A", "UNKNOWN", "9.01"])]


    if item_df.empty:
        return pd.DataFrame(columns=key_columns)

    # Tworzsenie nazwy cechy
    item_df["Item_Feature"] = "Has_Item_" + item_df["Item_Number"].str.replace(".","_",regex=False)

    item_df["Feature_Value"] = 1

    item_features = (
        item_df.pivot_table(
            index=key_columns,
            columns="Item_Feature",
            values="Feature_Value",
            aggfunc="max",
            fill_value=0,
        ).reset_index()
    )

    item_features.columns.name = None

    return item_features



def load_filing_metadata(metadata_file: Path) -> pd.DataFrame:
    """
    Wczytuje metadane SEC i redukuje je do jednego wiersza na filing
    """

    metadata_df = pd.read_csv(metadata_file)

    requiered_col = [
        "Ticker",
        "Filing_Date",
        "Acceptance_DateTime_ET",
        "Accession",
    ]

    missing_columns = [column for column in requiered_col if column not in metadata_df.columns]

    if missing_columns:
        raise ValueError(f"Brakuje kolumn w sec_document_metadata.csv: {missing_columns}")


    consistency_columns = ["Filing_Date", "Acceptance_DateTime_ET",]

    for column in consistency_columns:
        value_counts = metadata_df.groupby(["Ticker","Accession"])[column].nunique(dropna=False)
        
        invalid_filings = value_counts[value_counts > 1]

        if not invalid_filings.empty:
            raise ValueError(f"Jeden filing posiada więcej niż jedną wartość {column}:\n {invalid_filings}")



    # Jeden wiersz na filing jako obserwacja
    filing_metadata = metadata_df[requiered_col].drop_duplicates(subset=["Ticker", "Accession"]).reset_index(drop=True)


    return filing_metadata


# Agregacja datasetu
def aggregate_sentiment(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        raise ValueError("Pusty DataFrame")

    validate_dataframe(df)


    key_columns = ["Ticker", "Date", "Accession"]

    aggregated_rows: list[dict] = []

    grouped = df.groupby(key_columns, sort=True, dropna=False)

    for _, filing_df in grouped:

        aggregated_row = aggregate_single_filing(filing_df)
        aggregated_rows.append(aggregated_row)

    aggregated_df = pd.DataFrame(aggregated_rows)


    # Dodatnie cech z item
    item_features_df = create_item_features(df)

    if not item_features_df.empty:

        aggregated_df = aggregated_df.merge(item_features_df, on=key_columns, how="left")

    
    # Brakujące cechy item
    item_columns = [column for column in aggregated_df.columns if column.startswith("Has_Item_")]

    if item_columns:
        aggregated_df[item_columns] = aggregated_df[item_columns].fillna(0).astype(int)
        
    # Sortowanie wyników
    aggregated_df = aggregated_df.sort_values(by=key_columns).reset_index(drop=True)
    

    return aggregated_df



def main() -> None:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {INPUT_FILE}")

    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {METADATA_FILE}")

    
    # Wczytanie
    df_blocks = pd.read_csv(INPUT_FILE)
    df_metadata = load_filing_metadata(METADATA_FILE)

    logger.info("Liczba bloków wejściowych: %d", len(df_blocks),)

    # Agregacja
    df_features = aggregate_sentiment(df_blocks)

    logger.info("Liczba zagregowanych filingów: %d", len(df_features),)

    # Metadane
    df_features = df_features.merge(
        df_metadata[["Ticker", "Accession", "Filing_Date", "Acceptance_DateTime_ET",]],
        on=["Ticker", "Accession"],
        how="left",
        validate="one_to_one",
    )

    # Sprawdzenie braku metadanych
    missing_metadata = (df_features["Filing_Date"].isna())

    if missing_metadata.any():

        missing_filings = df_features.loc[missing_metadata, ["Ticker", "Accession"]]

        raise ValueError(f"Nie znaleziono metadanych SEC dla części filingów:\n {missing_filings.to_string(index=False)}")

    # Zgodność dat
    sentiment_dates = pd.to_datetime(df_features["Date"], errors="raise",).dt.date
    metadata_dates = pd.to_datetime(df_features["Filing_Date"], errors="raise",).dt.date

    date_mismatch = sentiment_dates != metadata_dates
    if date_mismatch.any():

        mismatched_filings = df_features.loc[date_mismatch, ["Ticker", "Accession", "Date", "Filing_Date"]]

        raise ValueError(f"Daty się nie zgadzają: {mismatched_filings.to_string(index=False)}")

    # Pozbycie sie dodatkowej kolumny
    df_features = df_features.drop(columns=["Date"])

    # Kolejnośćkolumn
    first_columns = ["Ticker", "Filing_Date", "Acceptance_DateTime_ET", "Accession"]

    remaining_columns = [column for column in df_features.columns if column not in first_columns]

    df_features = df_features[first_columns + remaining_columns]

    # Końcowy zapis
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(OUTPUT_FILE, index=False)

    logger.info("Zapisano %d filingów do %s", len(df_features), OUTPUT_FILE)

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format=("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    main()