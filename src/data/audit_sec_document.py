import os
import glob
import re
from collections import Counter, defaultdict


def extract_document_types(
    file_path: str,
) -> list[str]:
    """
    Zwraca wszystkie wartości <TYPE> znalezione
    w full-submission.txt.
    """

    try:
        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as file:
            raw_content = file.read()

    except Exception as error:
        print(
            f"Błąd odczytu {file_path}: {error}"
        )
        return []

    document_types = re.findall(
        r"<TYPE>\s*([^\r\n<]+)",
        raw_content,
        flags=re.IGNORECASE,
    )

    return [
        document_type.strip().upper()
        for document_type in document_types
    ]


def audit_sec_documents(
    base_input_dir: str,
    tickers: list[str],
) -> None:

    global_counter = Counter()

    types_by_ticker = defaultdict(
        Counter
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "AUDYT DOKUMENTÓW SEC"
    )

    print(
        "=" * 80
    )

    for ticker in tickers:

        search_pattern = os.path.join(
            base_input_dir,
            ticker,
            "8-K",
            "*",
            "full-submission.txt",
        )

        file_paths = sorted(
            glob.glob(
                search_pattern
            )
        )

        print(
            f"\n\n{'#' * 80}"
        )

        print(
            f"{ticker} | liczba filingów: "
            f"{len(file_paths)}"
        )

        print(
            f"{'#' * 80}"
        )

        for file_path in file_paths:

            accession_number = os.path.basename(
                os.path.dirname(
                    file_path
                )
            )

            document_types = (
                extract_document_types(
                    file_path
                )
            )

            global_counter.update(
                document_types
            )

            types_by_ticker[
                ticker
            ].update(
                document_types
            )

            print(
                f"\n{accession_number}"
            )

            for document_type in document_types:
                print(
                    f"  - {document_type}"
                )

    # ========================================================
    # PODSUMOWANIE GLOBALNE
    # ========================================================

    print(
        "\n\n"
        + "=" * 80
    )

    print(
        "PODSUMOWANIE GLOBALNE"
    )

    print(
        "=" * 80
    )

    for document_type, count in (
        global_counter.most_common()
    ):
        print(
            f"{document_type:<20} "
            f"{count}"
        )

    # ========================================================
    # PODSUMOWANIE PER TICKER
    # ========================================================

    print(
        "\n\n"
        + "=" * 80
    )

    print(
        "PODSUMOWANIE PER TICKER"
    )

    print(
        "=" * 80
    )

    for ticker in tickers:

        print(
            f"\n{ticker}"
        )

        for document_type, count in (
            types_by_ticker[
                ticker
            ].most_common()
        ):
            print(
                f"  {document_type:<20} "
                f"{count}"
            )


if __name__ == "__main__":

    INPUT_DIRECTORY = os.path.abspath(
        os.path.join(
            "data",
            "raw",
            "sec-edgar-filings",
        )
    )

    TARGET_TICKERS = [
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "GOOGL",
        "META",
        "TSLA",
        "AMD",
        "INTC",
        "AVGO",
        "NFLX",
        "ADBE",
        "QCOM",
    ]

    audit_sec_documents(
        base_input_dir=INPUT_DIRECTORY,
        tickers=TARGET_TICKERS,
    )