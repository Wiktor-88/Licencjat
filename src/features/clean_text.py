# Trzeci plik do czyszczenia dokmentów SEC (preprocessing przed NLP) 

import re
import logging

from pathlib import Path
from bs4 import BeautifulSoup


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Zdania po któreych usuwamy dalszą część komunikatu SEC
BOILERPLATE_PATTERNS = [
    re.compile(r"\bForward-Looking Statements\b", re.IGNORECASE),
    re.compile(r"\bForward Looking Statements\b", re.IGNORECASE),
    re.compile(r"\bCautionary Statement Regarding Forward-Looking Statements\b", re.IGNORECASE),
    re.compile(r"\bCautionary Note Regarding Forward-Looking Statements\b", re.IGNORECASE),
    re.compile(r"\bSafe Harbor Statement\b", re.IGNORECASE),
    re.compile(r"\bSafe Harbor\b", re.IGNORECASE),

    # Informacje organizacyjne
    re.compile(r"\bConference Call and Webcast Information\b", re.IGNORECASE),
    re.compile(r"\bConference Call Information\b", re.IGNORECASE),
    re.compile(r"\bWebcast Information\b", re.IGNORECASE),
]

# Na razie dla tych tylko firm - potem dodać więcej
COMPANY_NAMES = [
    "Apple",
    "Microsoft",
    "NVIDIA",
]

ABOUT_COMPANY_PATTERN = re.compile(rf"\bAbout ({'|'.join(map(re.escape, COMPANY_NAMES))})\b", re.IGNORECASE)

BOILERPLATE_PATTERNS.append(ABOUT_COMPANY_PATTERN)



def extract_relevant_documents(raw_content: str) -> list[dict]:
    """
    Wyciąga z całego dokumentu SEC tylko rzeczy ważne dla pipelinu, czyli:
    - 8-K,
    - załączniki EX-99.x
    Zwraca listę słowników
    """

    document_blocks = re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", raw_content, flags=re.IGNORECASE | re.DOTALL)

    relevant_documents = []

    for document_block in document_blocks:

        type_match = re.search(r"<TYPE>\s*([^\r\n<]+)", document_block, flags=re.IGNORECASE)

        if not type_match:
            continue

        # Zmiana wszytkich podobnych nazw na jedno zwykłe EX-99.x
        document_type = type_match.group(1).strip().upper()

        if document_type.startswith("EX-99"):
            ex99_match = re.match(r"^(EX-99(?:\.\d+)?)", document_type)

            if ex99_match:
                document_type = ex99_match.group(1)
      
        # Wybiernie 8-K i EX-99
        if not (document_type == "8-K" or document_type.startswith("EX-99")):
            continue

        text_match = re.search(r"<TEXT>(.*?)</TEXT>", document_block, flags=re.IGNORECASE | re.DOTALL)

        if not text_match:
            continue

        document_text = text_match.group(1)

        relevant_documents.append({
                "source_type": document_type,
                "raw_text": document_text
                })

    return relevant_documents



def extract_8k_items(text: str) -> str:
    """Usuwa boilerplate z 8-K"""

    # Usunięcie wszytkiego przed pierwszym item
    item_match = re.search(r"\bITEM\s+\d+\.\d+\.?", text, flags=re.IGNORECASE)

    if item_match:
        text = text[item_match.start():].strip()


    # Usunięcie sekcji signatures
    text = re.sub(r"\bSIGNATURES?\b.*$", "", text, flags=re.IGNORECASE | re.DOTALL)

    return text.strip()



def clean_exhibit_boilerplate(text: str) -> str:
    """
    Funkcja ta usuwa boilerplate z EX-99.x
    """

    earliest_match = None

    for pattern in BOILERPLATE_PATTERNS:

        match = pattern.search(text)
        if match:
           if earliest_match is None or match.start() < earliest_match:
               earliest_match = match.start()

    if earliest_match is not None:
        text = text[:earliest_match]

    text = re.sub(r"[_\-\s]{5,}$", "", text).strip()

    return text



def clean_html_text(html_text: str) -> str:
    """
    Czyści dokument HTML, czyli tutaj tekst SEC, z użyciem lxml
    """

    soup = BeautifulSoup(html_text, "lxml",)

    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()


    for tag in soup.find_all(
        lambda element:
            element.name
            and element.name.lower() == "ix:hidden"
    ):
        tag.decompose()

    # Nie usuwamy nagłówka tylko tabulatoory i wiele spacji, ale znak nowej lini zostaje
    clean_text = soup.get_text(separator="\n", strip=True)

    clean_text = re.sub(r"[ \t]", " ", clean_text)

    clean_text = re.sub(r"\n{2,}", "\n", clean_text)

    return clean_text



# Główne czyszsczenie
def clean_html_document(file_path: Path) -> list[dict[str, str]]:
    """
    Czyści wszystkie istotne dokumenty z submission SEC
    """

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            raw_content = file.read()

    except Exception as error:
        logging.error('błąd odczytu pliku: %s', file_path)

        return []


    relevant_documents = extract_relevant_documents(raw_content)

    cleaned_documents = []

    for document in relevant_documents:
        source_type = document["source_type"]
        raw_text = document["raw_text"]
        clean_text = clean_html_text(raw_text)

        # Rodzielamy 8K i EX99
        if source_type == "8-K":
            clean_text = extract_8k_items(clean_text)

        elif source_type.startswith("EX-99"):
            clean_text = clean_exhibit_boilerplate(clean_text)

        if not clean_text:
            continue

        cleaned_documents.append({
            "source_type": source_type,
            "text": clean_text,
        })

    return cleaned_documents


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",)

    # Szybki test na jednym z dokumentów
    test_path = PROJECT_ROOT / "data" / "raw" / "sec-edgar-filings" / "MSFT" / "8-K" / "0001193125-25-256310" / "full-submission.txt"
    

    if not test_path.exists():
        logger.error("Nie znaleziono pliku testowego: %s", test_path)
        raise SystemExit(1)

    print("\nTest dla pliku:\n"
          f"{test_path}\n")

    extracted_documents = clean_html_document(test_path)

    if not extracted_documents:
        print("Nie udało się wyodrębnić żadnych dokumentów")
        raise SystemExit(1)

    print('Udało sie wyodrebnic dokumenty')

    print(f"Liczba wyodrębnionych dokumentów: {len(extracted_documents)}")

    # Pokazujemy każdy dokument osobno
    for document_index, document in enumerate(extracted_documents, start=1):

        source_type = document["source_type"]

        text = document["text"]

        print(f"DOKUMENT {document_index}")

        print(f"Source Type: {source_type}")

        print(f"Długość tekstu: {len(text):,} znaków")