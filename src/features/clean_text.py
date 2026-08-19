import re
import os
import glob

from bs4 import BeautifulSoup


def extract_relevant_documents(raw_content: str) -> list[dict]:
    """
    Wyciąga z pełnego submission SEC tylko dokumenty istotne
    dla naszego pipeline'u:

    - główny dokument 8-K,
    - załączniki EX-99.x.

    Zwraca listę słowników:
        {
            "source_type": "...",
            "raw_text": "..."
        }
    """

    document_blocks = re.findall(
        r"<DOCUMENT>(.*?)</DOCUMENT>",
        raw_content,
        flags=re.IGNORECASE | re.DOTALL,
    )

    relevant_documents = []

    for document_block in document_blocks:

        type_match = re.search(
            r"<TYPE>\s*([^\r\n<]+)",
            document_block,
            flags=re.IGNORECASE,
        )

        if not type_match:
            continue

        document_type = type_match.group(1).strip()

        # ====================================================
        # NORMALIZACJA EX-99
        # ====================================================
        #
        # Przykład znaleziony w historycznych danych NVDA:
        #
        #   EX-99.1 PRESS RELEAS
        #
        # normalizujemy do:
        #
        #   EX-99.1
        #
        # Dzięki temu cały dalszy pipeline ma spójne
        # Source_Type.
        # ====================================================

        if document_type.upper().startswith("EX-99"):
            ex99_match = re.match(
                r"^(EX-99(?:\.\d+)?)",
                document_type,
                flags=re.IGNORECASE,
            )

            if ex99_match:
                document_type = ex99_match.group(1).upper()

        # ====================================================
        # INTERESUJĄ NAS TYLKO 8-K ORAZ EX-99.x
        # ====================================================

        if not (
            document_type.upper() == "8-K"
            or document_type.upper().startswith("EX-99")
        ):
            continue

        text_match = re.search(
            r"<TEXT>(.*?)</TEXT>",
            document_block,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if not text_match:
            continue

        document_text = text_match.group(1)

        relevant_documents.append(
            {
                "source_type": document_type,
                "raw_text": document_text,
            }
        )

    return relevant_documents


# ============================================================
# CZYSZCZENIE GŁÓWNEGO 8-K
# ============================================================

def extract_8k_items(text: str) -> str:
    """
    Usuwa standardowy boilerplate formularza 8-K
    znajdujący się przed pierwszą sekcją ITEM oraz
    formalną sekcję SIGNATURE na końcu.

    Dzięki temu do FinBERT-a trafia głównie właściwa,
    merytoryczna część raportu.
    """

    # ========================================================
    # 1. Usunięcie wszystkiego przed pierwszym ITEM
    # ========================================================

    item_match = re.search(
        r"\bITEM\s+\d+\.\d+\.?",
        text,
        flags=re.IGNORECASE,
    )

    if item_match:
        text = text[
            item_match.start():
        ].strip()

    # ========================================================
    # 2. Usunięcie sekcji SIGNATURE / SIGNATURES
    # ========================================================

    text = re.sub(
        r"\bSIGNATURES?\b.*$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    return text.strip()



# ============================================================
# CZYSZCZENIE ŚMIECI
# ============================================================

def clean_exhibit_boilerplate(
    text: str,
) -> str:
    """
    Usuwa końcowy, powtarzalny boilerplate z komunikatów EX-99.x,
    pozostawiając główną treść finansową.
    """

    boilerplate_markers = [
        # Disclaimer prawny
        r"\bForward-Looking Statements\b",
        r"\bForward Looking Statements\b",
        r"\bCautionary Statement Regarding Forward-Looking Statements\b",

        # Informacje o firmie / marketingowe stopki
        r"\bAbout Apple\b",
        r"\bAbout Microsoft\b",
        r"\bAbout NVIDIA\b",

        # Informacje organizacyjne
        r"\bConference Call and Webcast Information\b",
        r"\bConference Call Information\b",
        r"\bWebcast Information\b",

        # Kontakty
        r"\bInvestor Relations\b",
        r"\bMedia Relations\b",
        r"\bMedia Contact\b",
        r"\bFor further information, contact\b",
        r"\bFor more information, contact\b",
    ]

    earliest_match = None

    for pattern in boilerplate_markers:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            if (
                earliest_match is None
                or match.start() < earliest_match
            ):
                earliest_match = match.start()

    if earliest_match is not None:
        text = text[:earliest_match]

    text = re.sub(
        r"[_\-\s]{5,}$",
        "",
        text,
    ).strip()

    return text




# ============================================================
# CZYSZCZENIE HTML
# ============================================================

def clean_html_text(
    html_text: str,
) -> str:
    """
    Czyści pojedynczy dokument HTML/tekst SEC.
    """

    soup = BeautifulSoup(
        html_text,
        "html.parser",
    )

    # Usuwamy elementy techniczne.
    for tag in soup.find_all(
        ["script", "style", "noscript"]
    ):
        tag.decompose()

    # Usuwamy ukryte elementy Inline XBRL.
    for tag in soup.find_all(
        lambda element:
            element.name
            and element.name.lower() == "ix:hidden"
    ):
        tag.decompose()

    clean_text = soup.get_text(
        separator=" ",
        strip=True,
    )

    # Normalizacja whitespace.
    clean_text = re.sub(
        r"\s+",
        " ",
        clean_text,
    ).strip()

    return clean_text


# ============================================================
# GŁÓWNA FUNKCJA CZYSZCZĄCA SUBMISSION
# ============================================================

def clean_html_document(
    file_path: str,
) -> list[dict[str, str]]:
    """
    Czyści wszystkie istotne dokumenty z submission SEC.

    Przykładowy wynik:

    [
        {
            "source_type": "8-K",
            "text": "Item 2.02 ..."
        },
        {
            "source_type": "EX-99.1",
            "text": "Apple today announced..."
        }
    ]
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
            f"Błąd odczytu pliku "
            f"{file_path}: {error}"
        )

        return []

    relevant_documents = extract_relevant_documents(
        raw_content
    )

    cleaned_documents = []

    for document in relevant_documents:

        source_type = document[
            "source_type"
        ]

        raw_text = document[
            "raw_text"
        ]

        clean_text = clean_html_text(
            raw_text
        )

        # Tylko główny 8-K posiada strukturę ITEM,
        # którą chcemy specjalnie oczyścić.
        #
        # EX-99.x zostawiamy jako pełną treść,
        # ponieważ może zawierać earnings release,
        # guidance, CFO commentary itd.
        if source_type == "8-K":

            clean_text = extract_8k_items(
                clean_text
            )

        elif source_type.startswith("EX-99"):

            clean_text = clean_exhibit_boilerplate(
                clean_text
            )

        if not clean_text:
            continue

        cleaned_documents.append({
            "source_type": source_type,
            "text": clean_text,
        })

    return cleaned_documents


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    search_pattern = os.path.join(
        "data",
        "raw",
        "sec-edgar-filings",
        "MSFT",
        "8-K",
        "0001193125-25-256310",
        "full-submission.txt",
    )

    files = sorted(
        glob.glob(search_pattern)
    )

    print(
        f"Znaleziono plików: {len(files)}"
    )

    if not files:
        print(
            "Nie znaleziono żadnych raportów "
            "8-K dla NVDA."
        )

        raise SystemExit(1)

    test_path = search_pattern

    print(
        f"\nRozpoczynam test ekstrakcji dla pliku:\n"
        f"{test_path}\n"
    )

    extracted_documents = clean_html_document(
        test_path
    )

    if not extracted_documents:
        print(
            "Nie udało się wyodrębnić "
            "żadnych dokumentów."
        )

        raise SystemExit(1)

    print(
        "Proces zakończony sukcesem."
    )

    print(
        f"Liczba wyodrębnionych dokumentów: "
        f"{len(extracted_documents)}"
    )

    # Pokazujemy każdy dokument osobno.
    for document_index, document in enumerate(
        extracted_documents,
        start=1,
    ):

        source_type = document[
            "source_type"
        ]

        text = document[
            "text"
        ]

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"DOKUMENT {document_index}"
        )

        print(
            f"Source Type: {source_type}"
        )

        print(
            f"Długość tekstu: "
            f"{len(text):,} znaków"
        )

        print(
            "-" * 80
        )

        print("\n Początek")

        print(
            text[:1500]
        )

        print("\n Koniec")

        print(text[-1500:]  )

        print(
            "\n"
            + "=" * 80
        )