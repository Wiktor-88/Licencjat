import os                                               # Moduł służacy do interakcji z systemem plików i ścieżkami
import glob                                             # Moduł do wyszukiwania plików i folderów przy użyciu wzorców
import re                                               # Moduł do obsługi wyrażeń regularnych

from datetime import datetime                           # Moduł do obsługi dat i czasu
from zoneinfo import ZoneInfo                           # Moduł do obsługi stref czasowych
import pandas as pd


from features.clean_text import clean_html_document     # Moduł do oczyszczania dokumentów HTML z bazy SEC EDGAR


def extract_sec_metadata(
    file_path: str,
) -> tuple[str, str | None]:
    """
    Zwraca:
    - Filing_Date w formacie YYYY-MM-DD
    - Acceptance_DateTime_ET jako pełny timestamp SEC

    ACCEPTANCE-DATETIME jest traktowany jako czas
    America/New_York (ET).

    Jeżeli pełny timestamp nie istnieje, używamy
    FILED AS OF DATE jako fallback dla Filing_Date,
    ale nie wymyślamy godziny publikacji.
    """

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:
        raw_content = file.read()

    # ========================================================
    # 1. DOKŁADNY CZAS ACCEPTANCE
    # ========================================================

    acceptance_match = re.search(
        r"<ACCEPTANCE-DATETIME>\s*(\d{14})",
        raw_content,
        flags=re.IGNORECASE,
    )

    if acceptance_match:

        acceptance_raw = (
            acceptance_match.group(1)
        )

        acceptance_datetime = datetime.strptime(
            acceptance_raw,
            "%Y%m%d%H%M%S",
        )

        # SEC timestamps są w czasie Eastern Time.
        acceptance_datetime = (
            acceptance_datetime.replace(
                tzinfo=ZoneInfo(
                    "America/New_York"
                )
            )
        )

        filing_date = (
            acceptance_datetime
            .date()
            .isoformat()
        )

        acceptance_datetime_et = (
            acceptance_datetime.isoformat()
        )

        return (
            filing_date,
            acceptance_datetime_et,
        )

    # ========================================================
    # 2. FALLBACK: TYLKO DATA
    # ========================================================

    filed_match = re.search(
        r"FILED AS OF DATE:\s*(\d{8})",
        raw_content,
        flags=re.IGNORECASE,
    )

    if filed_match:

        filing_date = datetime.strptime(
            filed_match.group(1),
            "%Y%m%d",
        ).date().isoformat()

        return (
            filing_date,
            None,
        )

    raise ValueError(
        f"Nie znaleziono daty SEC w pliku: "
        f"{file_path}"
    )

def process_all_reports(
    base_input_dir: str,
    output_dir: str,
    tickers: list[str],
) -> None:

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    metadata_rows = []

    for ticker in tickers:

        print(
            f"\nPrzetwarzanie raportów: "
            f"{ticker}"
        )

        search_pattern = os.path.join(
            base_input_dir,
            ticker,
            "8-K",
            "**",
            "full-submission.txt",
        )

        file_paths = sorted(
            glob.glob(
                search_pattern,
                recursive=True,
            )
        )

        print(
            f"  Znaleziono raportów: "
            f"{len(file_paths)}"
        )

        for file_path in file_paths:

            # =================================================
            # METADANE
            # =================================================

            (
                filing_date,
                acceptance_datetime_et,
            ) = extract_sec_metadata(
                file_path
            )

            if filing_date is None:
                filing_date = "unknown-date"

            accession_number = os.path.basename(
                os.path.dirname(
                    file_path
                )
            )

            # =================================================
            # CZYSZCZENIE
            # =================================================

            cleaned_documents = (
                clean_html_document(
                    file_path
                )
            )

            if not cleaned_documents:
                print(
                    f"  Brak dokumentów dla: "
                    f"{accession_number}"
                )
                continue

            # =================================================
            # ZAPIS
            # =================================================

            for document in cleaned_documents:

                source_type = document[
                    "source_type"
                ]

                clean_text = document[
                    "text"
                ]

                if not clean_text.strip():
                    continue

                safe_source_type = (
                    source_type
                    .replace(".", "-")
                )

                output_filename = (
                    f"{ticker}_"
                    f"{filing_date}_"
                    f"{accession_number}_"
                    f"{safe_source_type}.txt"
                )

                output_path = os.path.join(
                    output_dir,
                    output_filename,
                )

                metadata_rows.append({
                    "Ticker": ticker,
                    "Filing_Date": filing_date,
                    "Acceptance_DateTime_ET":
                        acceptance_datetime_et,
                    "Accession": accession_number,
                    "Source_Type": source_type,
                    "Cleaned_File": output_filename,
                })

                with open(
                    output_path,
                    "w",
                    encoding="utf-8",
                ) as output_file:
                    output_file.write(
                        clean_text
                    )

                print(
                    f"  Zapisano: "
                    f"{output_filename}"
                )

    metadata_output_path = os.path.join(
        "data",
        "processed",
        "sec_document_metadata.csv",
    )

    metadata_df = pd.DataFrame(
        metadata_rows
    )

    metadata_df = metadata_df.sort_values(
        by=[
            "Ticker",
            "Filing_Date",
            "Accession",
            "Source_Type",
        ]
    ).reset_index(
        drop=True
    )

    metadata_df.to_csv(
        metadata_output_path,
        index=False,
    )

    print(
        f"\nMetadane SEC zapisano do:\n"
        f"{metadata_output_path}"
    )

if __name__ == '__main__':
    INPUT_DIRECTORY = os.path.abspath(os.path.join("data", "raw", "sec-edgar-filings"))
    OUTPUT_DIRECTORY = os.path.abspath(os.path.join("data", "processed", "cleaned_texts"))
    TARGET_TICKERS = ["AAPL", "MSFT", "NVDA"]
    
    process_all_reports(INPUT_DIRECTORY, OUTPUT_DIRECTORY, TARGET_TICKERS)