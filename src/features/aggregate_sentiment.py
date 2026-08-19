from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# ŚCIEŻKI
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sentiment_blocks.csv"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sec_document_metadata.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sentiment_filing_features.csv"
)


# ============================================================
# WYMAGANE KOLUMNY
# ============================================================

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


# ============================================================
# WALIDACJA
# ============================================================

def validate_dataframe(
    df: pd.DataFrame,
) -> None:

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Brakuje wymaganych kolumn w "
            "sentiment_blocks.csv:\n"
            + "\n".join(missing_columns)
        )


# ============================================================
# AGREGACJA JEDNEGO FILINGU
# ============================================================

def aggregate_single_filing(
    filing_df: pd.DataFrame,
) -> dict:

    ticker = filing_df["Ticker"].iloc[0]
    filing_date = filing_df["Date"].iloc[0]
    accession = filing_df["Accession"].iloc[0]

    # --------------------------------------------------------
    # PODZIAŁ NA 8-K I EX-99
    # --------------------------------------------------------

    blocks_8k = filing_df[
        filing_df["Source_Type"] == "8-K"
    ].copy()

    blocks_ex99 = filing_df[
        filing_df["Source_Type"].str.startswith(
            "EX-99",
            na=False,
        )
    ].copy()

    # --------------------------------------------------------
    # ITEMY Z GŁÓWNEGO 8-K
    # --------------------------------------------------------

    item_numbers = sorted(
        {
            str(item_number)
            for item_number in blocks_8k["Item_Number"]
            if pd.notna(item_number)
            and str(item_number) not in {
                "N/A",
                "UNKNOWN",
            }
        }
    )

    source_types = sorted(
        filing_df[
            "Source_Type"
        ].dropna().unique()
    )

    # --------------------------------------------------------
    # PODSTAWOWE METADANE
    # --------------------------------------------------------

    result = {
        "Ticker": ticker,
        "Date": filing_date,
        "Accession": accession,

        "Has_8K": int(
            not blocks_8k.empty
        ),

        "Has_EX99": int(
            not blocks_ex99.empty
        ),

        "Filing_Block_Count":
            len(filing_df),

        "EightK_Block_Count":
            len(blocks_8k),

        "EX99_Block_Count":
            len(blocks_ex99),

        "Item_Numbers":
            "|".join(item_numbers),

        "Source_Types":
            "|".join(source_types),
    }

    # ========================================================
    # BRAK EX-99
    # ========================================================
    #
    # To bardzo ważne:
    #
    # brak komunikatu EX-99 NIE oznacza neutralnego sentymentu.
    #
    # Dlatego wartości sentymentowe zostawiamy jako NaN.
    # ========================================================

    if blocks_ex99.empty:

        result.update({
            "Positive_Block_Count": 0,
            "Negative_Block_Count": 0,
            "Neutral_Block_Count": 0,

            "Positive_Ratio": np.nan,
            "Negative_Ratio": np.nan,
            "Neutral_Ratio": np.nan,

            "Mean_Positive": np.nan,
            "Mean_Negative": np.nan,
            "Mean_Neutral": np.nan,

            "Mean_Net_Sentiment": np.nan,
            "Median_Net_Sentiment": np.nan,
            "Min_Net_Sentiment": np.nan,
            "Max_Net_Sentiment": np.nan,

            "Max_Positive_Probability": np.nan,
            "Max_Negative_Probability": np.nan,

            "Most_Positive_Source_Type": None,
            "Most_Positive_Block_ID": np.nan,

            "Most_Negative_Source_Type": None,
            "Most_Negative_Block_ID": np.nan,
        })

        return result

    # ========================================================
    # LICZEBNOŚĆ KLAS
    # ========================================================

    labels = (
        blocks_ex99[
            "Predicted_Label"
        ]
        .astype(str)
        .str.lower()
    )

    positive_count = int(
        (labels == "positive").sum()
    )

    negative_count = int(
        (labels == "negative").sum()
    )

    neutral_count = int(
        (labels == "neutral").sum()
    )

    sentiment_block_count = len(
        blocks_ex99
    )

    # ========================================================
    # NAJBARDZIEJ POZYTYWNY BLOK
    # ========================================================

    most_positive_index = (
        blocks_ex99[
            "Prob_Positive"
        ].idxmax()
    )

    most_positive_row = (
        blocks_ex99.loc[
            most_positive_index
        ]
    )

    # ========================================================
    # NAJBARDZIEJ NEGATYWNY BLOK
    # ========================================================

    most_negative_index = (
        blocks_ex99[
            "Prob_Negative"
        ].idxmax()
    )

    most_negative_row = (
        blocks_ex99.loc[
            most_negative_index
        ]
    )

    # ========================================================
    # CECHY SENTYMENTU
    # ========================================================

    result.update({

        # ----------------------------------------------------
        # LICZEBNOŚCI
        # ----------------------------------------------------

        "Positive_Block_Count":
            positive_count,

        "Negative_Block_Count":
            negative_count,

        "Neutral_Block_Count":
            neutral_count,

        # ----------------------------------------------------
        # UDZIAŁY KLAS
        # ----------------------------------------------------

        "Positive_Ratio":
            positive_count
            / sentiment_block_count,

        "Negative_Ratio":
            negative_count
            / sentiment_block_count,

        "Neutral_Ratio":
            neutral_count
            / sentiment_block_count,

        # ----------------------------------------------------
        # ŚREDNIE PRAWDOPODOBIEŃSTWA FINBERT
        # ----------------------------------------------------

        "Mean_Positive":
            blocks_ex99[
                "Prob_Positive"
            ].mean(),

        "Mean_Negative":
            blocks_ex99[
                "Prob_Negative"
            ].mean(),

        "Mean_Neutral":
            blocks_ex99[
                "Prob_Neutral"
            ].mean(),

        # ----------------------------------------------------
        # NET SENTIMENT
        # ----------------------------------------------------

        "Mean_Net_Sentiment":
            blocks_ex99[
                "Net_Sentiment"
            ].mean(),

        "Median_Net_Sentiment":
            blocks_ex99[
                "Net_Sentiment"
            ].median(),

        "Min_Net_Sentiment":
            blocks_ex99[
                "Net_Sentiment"
            ].min(),

        "Max_Net_Sentiment":
            blocks_ex99[
                "Net_Sentiment"
            ].max(),

        # ----------------------------------------------------
        # EKSTREMALNY SENTYMENT
        # ----------------------------------------------------

        "Max_Positive_Probability":
            blocks_ex99[
                "Prob_Positive"
            ].max(),

        "Max_Negative_Probability":
            blocks_ex99[
                "Prob_Negative"
            ].max(),

        # ----------------------------------------------------
        # TRACEABILITY / XAI
        # ----------------------------------------------------

        "Most_Positive_Source_Type":
            most_positive_row[
                "Source_Type"
            ],

        "Most_Positive_Block_ID":
            int(
                most_positive_row[
                    "Block_ID"
                ]
            ),

        "Most_Negative_Source_Type":
            most_negative_row[
                "Source_Type"
            ],

        "Most_Negative_Block_ID":
            int(
                most_negative_row[
                    "Block_ID"
                ]
            ),
    })

    return result


# ============================================================
# CECHY ITEMÓW 8-K
# ============================================================

def create_item_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    key_columns = [
        "Ticker",
        "Date",
        "Accession",
    ]

    item_df = df[
        df["Source_Type"] == "8-K"
    ].copy()

    item_df = item_df[
        item_df[
            "Item_Number"
        ].notna()
    ]

    item_df = item_df[
        ~item_df[
            "Item_Number"
        ].astype(str).isin(
            [
                "N/A",
                "UNKNOWN",
            ]
        )
    ]

    # Item 9.01 to techniczna sekcja
    # Financial Statements and Exhibits.
    # Zachowujemy ją w Item_Numbers,
    # ale nie tworzymy z niej cechy modelowej.
    item_df = item_df[
        item_df[
            "Item_Number"
        ].astype(str) != "9.01"
    ]

    if item_df.empty:
        return pd.DataFrame(
            columns=key_columns
        )

    # Przykład:
    #
    # 2.02 -> Has_Item_2_02
    # 5.02 -> Has_Item_5_02

    item_df[
        "Item_Feature"
    ] = (
        "Has_Item_"
        + item_df[
            "Item_Number"
        ]
        .astype(str)
        .str.replace(
            ".",
            "_",
            regex=False,
        )
    )

    item_df[
        "Feature_Value"
    ] = 1

    item_features = (
        item_df.pivot_table(
            index=key_columns,
            columns="Item_Feature",
            values="Feature_Value",
            aggfunc="max",
            fill_value=0,
        )
        .reset_index()
    )

    item_features.columns.name = None

    return item_features



def load_filing_metadata(
    metadata_file: Path,
) -> pd.DataFrame:
    """
    Wczytuje metadane SEC i redukuje je
    do jednego wiersza na filing.

    Jeden accession może posiadać kilka dokumentów:
    8-K, EX-99.1, EX-99.2 itd.,
    ale wszystkie mają ten sam Acceptance_DateTime_ET.
    """

    metadata_df = pd.read_csv(
        metadata_file
    )

    required_columns = [
        "Ticker",
        "Filing_Date",
        "Acceptance_DateTime_ET",
        "Accession",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in metadata_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Brakuje kolumn w "
            "sec_document_metadata.csv: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # SPRAWDZENIE SPÓJNOŚCI TIMESTAMPÓW
    # --------------------------------------------------------

    timestamp_counts = (
        metadata_df.groupby(
            [
                "Ticker",
                "Accession",
            ]
        )[
            "Acceptance_DateTime_ET"
        ]
        .nunique(
            dropna=False
        )
    )

    invalid_filings = (
        timestamp_counts[
            timestamp_counts > 1
        ]
    )

    if not invalid_filings.empty:
        raise ValueError(
            "Jeden filing posiada więcej niż jeden "
            "Acceptance_DateTime_ET:\n"
            f"{invalid_filings}"
        )

    # --------------------------------------------------------
    # JEDEN WIERSZ NA FILING
    # --------------------------------------------------------

    filing_metadata = (
        metadata_df[
            required_columns
        ]
        .drop_duplicates(
            subset=[
                "Ticker",
                "Accession",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return filing_metadata


# ============================================================
# AGREGACJA CAŁEGO DATASETU
# ============================================================

def aggregate_sentiment(
    df: pd.DataFrame,
) -> pd.DataFrame:

    key_columns = [
        "Ticker",
        "Date",
        "Accession",
    ]

    aggregated_rows = []

    grouped = df.groupby(
        key_columns,
        sort=True,
        dropna=False,
    )

    for _, filing_df in grouped:

        aggregated_row = (
            aggregate_single_filing(
                filing_df
            )
        )

        aggregated_rows.append(
            aggregated_row
        )

    aggregated_df = pd.DataFrame(
        aggregated_rows
    )

    




    # ========================================================
    # DODAJEMY ITEM FEATURES
    # ========================================================

    item_features_df = (
        create_item_features(
            df
        )
    )

    if not item_features_df.empty:

        aggregated_df = (
            aggregated_df.merge(
                item_features_df,
                on=key_columns,
                how="left",
            )
        )

    # ========================================================
    # BRAKUJĄCE ITEM FEATURES -> 0
    # ========================================================

    item_columns = [
        column
        for column
        in aggregated_df.columns
        if column.startswith(
            "Has_Item_"
        )
    ]

    for column in item_columns:

        aggregated_df[column] = (
            aggregated_df[
                column
            ]
            .fillna(0)
            .astype(int)
        )

    aggregated_df = (
        aggregated_df.sort_values(
            by=[
                "Ticker",
                "Date",
                "Accession",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return aggregated_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 70
    )

    print(
        "AGREGACJA SENTYMENTU FINBERT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nPlik wejściowy:\n{INPUT_FILE}"
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku:\n"
            f"{INPUT_FILE}"
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku:\n"
            f"{METADATA_FILE}"
        )

    # ========================================================
    # WCZYTANIE
    # ========================================================

    df_blocks = pd.read_csv(
        INPUT_FILE
    )

    df_metadata = load_filing_metadata(
        METADATA_FILE
    )

    validate_dataframe(
        df_blocks
    )

    print(
        f"\nLiczba bloków wejściowych: "
        f"{len(df_blocks)}"
    )

    print(
        f"Liczba unikalnych filingów: "
        f"{df_blocks[
            ['Ticker', 'Date', 'Accession']
        ].drop_duplicates().shape[0]}"
    )

    # ========================================================
    # AGREGACJA
    # ========================================================

    df_features = aggregate_sentiment(
        df_blocks
    )

    # ============================================================
    # DODANIE DOKŁADNEGO CZASU PUBLIKACJI SEC
    # ============================================================

    df_features = df_features.merge(
        df_metadata[
            [
                "Ticker",
                "Accession",
                "Filing_Date",
                "Acceptance_DateTime_ET",
            ]
        ],
        on=[
            "Ticker",
            "Accession",
        ],
        how="left",
        validate="one_to_one",
    )





    # ============================================================
    # SPRAWDZENIE ZGODNOŚCI DAT
    # ============================================================

    date_mismatch = (
        pd.to_datetime(
            df_features["Date"]
        ).dt.date
        !=
        pd.to_datetime(
            df_features["Filing_Date"]
        ).dt.date
    )

    if date_mismatch.any():
        raise ValueError(
            "Date z sentiment_blocks.csv "
            "nie zgadza się z Filing_Date "
            "z metadanych SEC."
        )

    # Stara kolumna Date nie jest już potrzebna.
    df_features = df_features.drop(
        columns=[
            "Date",
        ]
    )

    # ============================================================
    # KOLEJNOŚĆ KOLUMN
    # ============================================================

    first_columns = [
        "Ticker",
        "Filing_Date",
        "Acceptance_DateTime_ET",
        "Accession",
    ]

    remaining_columns = [
        column
        for column in df_features.columns
        if column not in first_columns
    ]

    df_features = df_features[
        first_columns
        + remaining_columns
    ]

    # ========================================================
    # ZAPIS
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df_features.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # PODGLĄD
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "WYNIK AGREGACJI"
    )

    print(
        "=" * 70
    )

    columns_to_show = [
        "Ticker",
        "Filing_Date",
        "Acceptance_DateTime_ET",
        "Accession",
        "Has_8K",
        "Has_EX99",
        "EX99_Block_Count",
        "Positive_Ratio",
        "Negative_Ratio",
        "Neutral_Ratio",
        "Mean_Net_Sentiment",
        "Max_Positive_Probability",
        "Max_Negative_Probability",
        "Item_Numbers",
    ]

    print(
        df_features[
            columns_to_show
        ].to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"Liczba wynikowych filingów: "
        f"{len(df_features)}"
    )

    print(
        f"\nWyniki zapisano do:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        "=" * 70
    )