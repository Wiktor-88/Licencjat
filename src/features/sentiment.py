# Czwarty plik odpowiada za dużo rzeczy, w tym  analze sentymentów
# Zostal on podzielony na 12 różnych częśći



# Czesc I - importy
import time
import re
import logging
import hashlib
import pandas as pd
import nltk
import torch

from pathlib import Path
from nltk.tokenize import sent_tokenize
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


# Część II - konfguracja
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "cleaned_texts"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "sentiment_blocks.csv"
CHECKPOINT_FILE = PROJECT_ROOT / "data" / "processed" / "sentiment_checkpoint.csv"

MODEL_NAME = "yiyanghkust/finbert-tone"
PIPELINE_VERSION = 2

# FinBERT ma limit 512 tokenów, wiec podzielimy tekst na bloki po 450 tokenów
MAX_TOKENS_PER_BLOCK = 450

# Ile zdań jest przenoszonych z poprzedniego bloku - aby zachowac kontekst
SENTENCE_OVERLAP = 0

# FinBERT będzie analizował do 32 bloków na raz na GPU
BATCH_SIZE = 32

# Co ile plików jest checkpoint
CHECKPOINT_EVERY = 50

# CPU/Cuda
DEVICE = 0 if torch.cuda.is_available() else -1

# Limit FinBerta
MODEL_MAX_LENGTH = 512

EXPECTED_LABELS = {"positive", "negative", "neutral"}


# Czesc III - NLTK
def ensure_nltk_resource(resource_path: str, download_name: str) -> None:
    """
    Sprawdza czy dany zasób NLTK jest dostępny, gdy nie ma to pobiera
    """

    try:
        nltk.data.find(resource_path)
        return

    except LookupError:
        logger.info("Brak zasobu NLTK %s, rozpoczynam pobieranie.", download_name)

    success = nltk.download(download_name, quiet=True)

    if not success:
        raise RuntimeError(f"Nie udało się pobrać zasobu NLTK: {download_name}")


# Wywołanie powyżesz funkcji, aby nie było problemu przy imporcie tego pliku
def ensure_nltk_resources() -> None:
    ensure_nltk_resource("tokenizers/punkt", "punkt")

    ensure_nltk_resource("tokenizers/punkt_tab/english", "punkt_tab")



# Czesc IV - wydobycie meatadanych
def parse_file_metadata(file_path: Path) -> tuple[str, str, str, str]:
    """
    Wyciąga z nazwy:
    - ticker
    - datę filing
    - accession number
    - typ dokumentu
    """

    filename_without_extension = file_path.stem

    # dokładnie 3 podziały, wiec zwraca 4 elementy
    parts = filename_without_extension.split("_", 3)

    if len(parts) != 4:
        raise ValueError(f"Nieprawidłowy format nazwy pliku: {file_path.name}")

    (ticker, filing_date, accession_number, source_type) = parts

    # zmiana formatu na EX-99.x
    if source_type.startswith("EX-99-"):
        source_type = source_type.replace("EX-99-", "EX-99.", 1)

    return (ticker, filing_date, accession_number, source_type)


# Czesc V - dzielenie dokumentów 8K
def split_into_8k_items(text: str) -> list[tuple[str, str]]:
    """
    Dzieli oczyszczony raport 8-K na osobne sekcje ITEM, dzięki temu każdy późniejszy
    blok z FinBERTa można jednoznacznie powiązać z daną sekcją raportu

    Zwraca liste krotek: (Item_number, tekst)
    """

    # Szukanie nagłówków typu: Item 1.01, ITEM 2.02.  czy item 5.02
    item_matches = list(
        re.finditer(r"^\s*ITEM\s+(\d+\.\d+)\.?", text, flags=(re.IGNORECASE | re.MULTILINE)))


    # fallback dla bezpieczeństwa (teoretycznie juz powinno działac w clean_text.py)
    if not item_matches:
        return [("UNKNOWN", text)]


    item_sections : list[tuple[str, str]] = []

    for index, match in enumerate(item_matches):
        item_number = match.group(1)

        section_start = match.start()

        # Koniec sekcji to początek następnego ITEM
        if index + 1 < len(item_matches):
            section_end = item_matches[index + 1].start()

        # Chyba ze dla ostatniego - tam nie istnieje item_matches[index + 1]
        else:
            section_end = len(text)

        # Wyciecie tekstu
        section_text = text[section_start:section_end].strip()

        # Dodjaemy kiedy cos wogóle zostało
        if section_text:
            item_sections.append((item_number, section_text))

    return item_sections



# Czesc VI - tworzenie bloków
def create_sentence_blocks(
    text: str,
    tokenizer,
    max_tokens: int = MAX_TOKENS_PER_BLOCK,
    overlap_sentences: int = SENTENCE_OVERLAP,
) -> list[str]:
    """
    Dzieli dokument na bloki składające się z pełnych zdań
    Każdy blok ma maksymalnie około 'max_tokens' tokenów FinBERT-a 
    (troche mniej aby na pewno sie zmiesiło w limicie)

    Między kolejnymi blokami zachowywany jest overlap kilku zdań (obecnie 0)
    """

    sentences = sent_tokenize(text, language="english")

    blocks: list[str] = []
    current_block: list[str] = []
    current_lengths: list[int] = []
    current_token_count = 0

    for sentence in sentences:

        # Tokenizujemy zdanie tokenizerem FinBERT-a, a nie tokenizerem NLTK
        sentence_token_ids = tokenizer.encode(sentence, add_special_tokens=False)

        sentence_length = len(sentence_token_ids)

        # Przypadek I - jedno bardzo dlugie zdanie
        if sentence_length > max_tokens:

            # Zapis aktualnie budowanego bloku
            if current_block:
                blocks.append(" ".join(current_block))

                current_block = []
                current_lengths = []
                current_token_count = 0

            # Cięcie tego zdania
            for start in range(0, sentence_length, max_tokens):
                token_fragment = sentence_token_ids[start:start + max_tokens]

                fragment_text = tokenizer.decode(token_fragment, skip_special_tokens=True)

                if fragment_text.strip():
                    blocks.append(fragment_text)

            continue

        
        # Przypadek II - zwyczajne zdanie mieszczące sie w limicie
        if current_token_count + sentence_length <= max_tokens:

            current_block.append(sentence)
            current_lengths.append(sentence_length)
            current_token_count += sentence_length

            continue

        
        # Przypadek III - dodanie zdania przekracza limit

        # Najpierw zapis
        if current_block:
            blocks.append(" ".join(current_block))

        # Overlap zdan 
        if overlap_sentences > 0:
            overlap = current_block[ -overlap_sentences:]

            overlap_lengths = current_lengths[ -overlap_sentences:]
        else:
            overlap = []
            overlap_lengths = []

        overlap_token_count = sum(overlap_lengths)

        # jeżeli overlap i nowe zdanie nadal przekracza 450 to zmiejszamy overlap
        while (overlap and overlap_token_count + sentence_length > max_tokens):
            removed_length = overlap_lengths.pop(0)
            overlap.pop(0)

            overlap_token_count -= removed_length


        current_block = overlap + [sentence]

        current_lengths = (overlap_lengths + [sentence_length])

        current_token_count = overlap_token_count + sentence_length


    
    # Przyapdek IV - ostatni blok
    if current_block:
        blocks.append(" ".join(current_block))

    return blocks


# Częsci VII - Analiza sentymentów
def load_finbert():
    """
    Ładuje potrzbne rzeczy: tokenzier, FinBert i pipeline
    """

    if DEVICE == 0:
        logger.info("FinBERT działa na GPU: %s", torch.cuda.get_device_name(0))
    else:
        logger.info("FinBERT działa na CPU")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    
    model_labels = {str(label).lower() for label in model.config.id2label.values()}

    if model_labels != EXPECTED_LABELS:
        raise ValueError(
            "Nieoczekiwane klasy modelu FinBERT. "
            f"Otrzymano: {sorted(model_labels)}, "
            f"oczekiwano: {sorted(EXPECTED_LABELS)}."
        )

    sentiment_pipeline = pipeline(
        task="sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        top_k=None,
    )

    logger.info("FinBERT został załadowany")

    return tokenizer, sentiment_pipeline


# Wybieranie plików
def select_input_files(input_dir: Path,
                       limit_files: int | None = None,
                       test_files_per_ticker: int | None = None) -> list[Path]:
    """
    Zwraca posortowaną listę plików wejściowych

    Opcjonalnie ogranicza liczbę plików globalnie albo wybiera określoną liczbę plików per ticker
    """

    if not input_dir.exists():
        raise FileNotFoundError(f"Nie istnieje katalog wejściowy: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Ścieżka wejściowa nie jest katalogiem: {input_dir}")

    if (limit_files is not None and test_files_per_ticker is not None):
        raise ValueError('Nie można jednocześnie używać limit_files i test_files_per_ticker')

    if (limit_files is not None and limit_files <= 0):
        raise ValueError("limit_files musi być większe od 0")

    if (test_files_per_ticker is not None and test_files_per_ticker <= 0):
        raise ValueError("test_files_per_ticker musi być większe od 0")

    file_paths = sorted(input_dir.glob("*.txt"))

    if test_files_per_ticker is not None:

        files_by_ticker: dict[str, list[Path]] = {}

        for file_path in file_paths:

            try:
                ticker, _, _, _ = parse_file_metadata(file_path)

            except ValueError as error:
                logger.warning("Pominięto plik %s: %s", file_path.name, error)
                continue

            files_by_ticker.setdefault(ticker, []).append(file_path)

        selected_files: list[Path] = []

        for ticker in sorted(files_by_ticker):
            selected_files.extend(files_by_ticker[ticker][:test_files_per_ticker])

        return selected_files

    if limit_files is not None:
        return file_paths[:limit_files]

    return file_paths



# Tworzenie bloków na dokumentach
def create_document_blocks(text: str, source_type: str, tokenizer,) -> list[dict]:
    """
    Dzieli dokument na logiczne sekcje, a następnie na bloki wejściowe FinBERT-a
    """

    if source_type == "8-K":

        item_sections = split_into_8k_items(
            text
        )

    elif source_type.startswith("EX-99"):

        item_sections = [
            ("N/A", text)
        ]

    else:
        raise ValueError(
            "Nieobsługiwany typ dokumentu: "
            f"{source_type}"
        )

    blocks_with_metadata: list[dict] = []

    global_block_id = 0

    for item_number, item_text in item_sections:

        item_blocks = create_sentence_blocks(
            text=item_text,
            tokenizer=tokenizer,
            max_tokens=MAX_TOKENS_PER_BLOCK,
            overlap_sentences=SENTENCE_OVERLAP,
        )

        for item_block_id, block_text in enumerate(
            item_blocks
        ):

            blocks_with_metadata.append({
                "Block_ID":
                    global_block_id,

                "Item_Number":
                    item_number,

                "Item_Block_ID":
                    item_block_id,

                "Text":
                    block_text,

                "Token_Count":
                    len(tokenizer.encode(block_text, add_special_tokens=False)),
            })

            global_block_id += 1

    return blocks_with_metadata


# ============================================================
# INFERENCJA FINBERT
# ============================================================

def predict_sentiment(
    blocks: list[str],
    sentiment_pipeline,
):
    """
    Przeprowadza batched inference FinBERT-a
    dla wszystkich bloków jednego dokumentu.
    """

    if not blocks:
        return []

    try:
        return sentiment_pipeline(
            blocks,
            truncation=True,
            max_length=MODEL_MAX_LENGTH,
            batch_size=BATCH_SIZE,
        )

    except RuntimeError as error:

        if (
            "out of memory"
            in str(error).lower()
        ):

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.exception(
                "Brak pamięci GPU podczas inferencji "
                "FinBERT. Aktualny BATCH_SIZE=%d.",
                BATCH_SIZE,
            )

        else:
            logger.exception(
                "RuntimeError podczas inferencji "
                "FinBERT."
            )

        raise


# ============================================================
# INTERPRETACJA PREDYKCJI
# ============================================================

def parse_sentiment_prediction(
    predictions: list[dict],
) -> dict:
    """
    Zamienia wynik jednego bloku FinBERT-a
    na cechy sentymentu.
    """

    if not predictions:
        raise ValueError(
            "FinBERT zwrócił pustą predykcję."
        )

    scores = {
        item["label"].lower():
            float(item["score"])
        for item in predictions
    }

    received_labels = set(
        scores
    )

    if received_labels != EXPECTED_LABELS:

        missing_labels = (
            EXPECTED_LABELS
            - received_labels
        )

        unexpected_labels = (
            received_labels
            - EXPECTED_LABELS
        )

        raise ValueError(
            "Niepoprawny zestaw klas FinBERT. "
            f"Brakujące: {sorted(missing_labels)}, "
            f"nieoczekiwane: "
            f"{sorted(unexpected_labels)}."
        )

    prob_positive = (
        scores["positive"]
    )

    prob_negative = (
        scores["negative"]
    )

    prob_neutral = (
        scores["neutral"]
    )

    net_sentiment = (
        prob_positive
        - prob_negative
    )

    predicted_label = max(
        scores,
        key=lambda label: scores[label],
    )

    confidence = (
        scores[predicted_label]
    )

    return {
        "Prob_Positive":
            prob_positive,

        "Prob_Negative":
            prob_negative,

        "Prob_Neutral":
            prob_neutral,

        "Net_Sentiment":
            net_sentiment,

        "Predicted_Label":
            predicted_label.capitalize(),

        "Confidence":
            confidence,
    }


# ============================================================
# TWORZENIE WIERSZA WYNIKOWEGO
# ============================================================

def create_result_row(
    ticker: str,
    filing_date: str,
    accession_number: str,
    source_type: str,
    filename: str,
    text_sha256: str,
    block_metadata: dict,
    predictions: list[dict],
) -> dict:
    """
    Łączy metadane bloku z wynikami FinBERT-a.
    """

    sentiment_features = (
        parse_sentiment_prediction(
            predictions
        )
    )

    result = {
        "Ticker":
            ticker,

        "Date":
            filing_date,

        "Accession":
            accession_number,

        "Source_Type":
            source_type,

        "File":
            filename,

        "Cleaned_Text_SHA256":
            text_sha256,

        "Sentiment_Model":
            MODEL_NAME,

        "Sentiment_Pipeline_Version":
            PIPELINE_VERSION,

        "Item_Number":
            block_metadata["Item_Number"],

        "Block_ID":
            block_metadata["Block_ID"],

        "Item_Block_ID":
            block_metadata["Item_Block_ID"],

        "Text_Snippet":
            block_metadata["Text"],

        "Token_Count":
            block_metadata["Token_Count"],
    }

    result.update(
        sentiment_features
    )

    return result


# ============================================================
# ANALIZA JEDNEGO PLIKU
# ============================================================

def process_sentiment_file(
    file_path: Path,
    tokenizer,
    sentiment_pipeline,
) -> list[dict]:
    """
    Przeprowadza pełną analizę sentymentu
    jednego oczyszczonego dokumentu SEC.
    """

    file_start_time = (
        time.perf_counter()
    )

    (
        ticker,
        filing_date,
        accession_number,
        source_type,
    ) = parse_file_metadata(
        file_path
    )

    text = file_path.read_text(
        encoding="utf-8"
    )
    text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()

    if not text.strip():

        logger.warning(
            "Pusty plik: %s",
            file_path.name,
        )

        return []

    # --------------------------------------------------------
    # CHUNKING
    # --------------------------------------------------------

    chunk_start = (
        time.perf_counter()
    )

    blocks_with_metadata = (
        create_document_blocks(
            text=text,
            source_type=source_type,
            tokenizer=tokenizer,
        )
    )

    chunk_time = (
        time.perf_counter()
        - chunk_start
    )

    blocks = [
        block["Text"]
        for block in blocks_with_metadata
    ]

    if not blocks:

        logger.warning(
            "Nie utworzono bloków dla %s",
            file_path.name,
        )

        return []

    # --------------------------------------------------------
    # FINBERT
    # --------------------------------------------------------

    finbert_start = (
        time.perf_counter()
    )

    all_predictions = predict_sentiment(
        blocks=blocks,
        sentiment_pipeline=sentiment_pipeline,
    )

    finbert_time = (
        time.perf_counter()
        - finbert_start
    )

    if (
        len(all_predictions)
        != len(blocks_with_metadata)
    ):
        raise ValueError(
            "Liczba predykcji FinBERT "
            "nie zgadza się z liczbą bloków. "
            f"Predykcje={len(all_predictions)}, "
            f"bloki={len(blocks_with_metadata)}."
        )

    # --------------------------------------------------------
    # WYNIKI
    # --------------------------------------------------------

    file_results: list[dict] = []

    for block_metadata, predictions in zip(
        blocks_with_metadata,
        all_predictions,
    ):

        result_row = create_result_row(
            ticker=ticker,
            filing_date=filing_date,
            accession_number=accession_number,
            source_type=source_type,
            filename=file_path.name,
            text_sha256=text_sha256,
            block_metadata=block_metadata,
            predictions=predictions,
        )

        file_results.append(
            result_row
        )

    # --------------------------------------------------------
    # STATYSTYKI CZASOWE
    # --------------------------------------------------------

    file_time = (
        time.perf_counter()
        - file_start_time
    )

    blocks_per_second = (
        len(blocks) / finbert_time
        if finbert_time > 0
        else 0.0
    )

    logger.info(
        "%s | bloki=%d | chunking=%.2fs | "
        "FinBERT=%.2fs | %.2f bloków/s | "
        "całość=%.2fs",
        file_path.name,
        len(blocks),
        chunk_time,
        finbert_time,
        blocks_per_second,
        file_time,
    )

    return file_results


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    results: list[dict],
    checkpoint_path: Path,
) -> None:
    """
    Zapisuje dotychczasowe wyniki analizy.
    """

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_df = pd.DataFrame(
        results
    )

    checkpoint_df.to_csv(
        checkpoint_path,
        index=False,
    )

    logger.info(
        "Zapisano checkpoint: %s",
        checkpoint_path,
    )


# ============================================================
# ANALIZA CAŁEGO KATALOGU
# ============================================================

def analyze_sentiment_directory(
    input_dir: Path,
    checkpoint_path: Path | None = None,
    limit_files: int | None = None,
    test_files_per_ticker: int | None = None,
) -> pd.DataFrame:
    """
    Przeprowadza analizę sentymentu
    wszystkich wybranych dokumentów SEC.
    """

    file_paths = select_input_files(
        input_dir=input_dir,
        limit_files=limit_files,
        test_files_per_ticker=test_files_per_ticker,
    )
    number_of_files = len(file_paths)
    logger.info("Liczba plików do analizy: %d", number_of_files)

    if number_of_files == 0:
        logger.warning("Nie znaleziono plików do analizy.")
        return pd.DataFrame()

    results: list[dict] = []
    file_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in file_paths}
    completed_files: set[str] = set()

    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = pd.read_csv(checkpoint_path)
        required = {"File", "Cleaned_Text_SHA256", "Sentiment_Model", "Sentiment_Pipeline_Version", "Token_Count"}
        if required.issubset(checkpoint.columns):
            checkpoint = checkpoint.loc[
                checkpoint["File"].isin(file_hashes)
                & checkpoint["Sentiment_Model"].eq(MODEL_NAME)
                & checkpoint["Sentiment_Pipeline_Version"].eq(PIPELINE_VERSION)].copy()

            for filename, file_df in checkpoint.groupby("File"):
                expected_hash = file_hashes.get(str(filename))
                if expected_hash and file_df["Cleaned_Text_SHA256"].eq(expected_hash).all():
                    completed_files.add(str(filename))
                    results.extend(file_df.to_dict("records"))
            logger.info("Wznowienie z checkpointu: %d/%d plików jest gotowych.", len(completed_files), number_of_files)
        else:
            logger.info("Checkpoint ma starszy format, dlatego analiza zostanie wykonana od początku.")

    pending_files = [path for path in file_paths if path.name not in completed_files]
    if not pending_files:
        logger.info("Wszystkie pliki są już zapisane w aktualnym checkpoincie.")
        return pd.DataFrame(results).sort_values(["Ticker", "Date", "Accession", "Source_Type", "Block_ID"]).reset_index(drop=True)

    ensure_nltk_resources()
    tokenizer, sentiment_pipeline = load_finbert()
    total_start_time = time.perf_counter()

    for file_index, file_path in enumerate(pending_files, start=1):
        logger.info("Przetwarzanie %d/%d pozostałych: %s", file_index, len(pending_files), file_path.name)

        file_results = process_sentiment_file(
            file_path=file_path,
            tokenizer=tokenizer,
            sentiment_pipeline=sentiment_pipeline)
        results.extend(file_results)

        if checkpoint_path is not None and file_index % CHECKPOINT_EVERY == 0:
            save_checkpoint(results=results, checkpoint_path=checkpoint_path)

    total_time = time.perf_counter() - total_start_time

    if checkpoint_path is not None:
        save_checkpoint(results=results, checkpoint_path=checkpoint_path)

    logger.info("Analiza zakończona | pliki=%d | bloki=%d | czas=%.2fs", number_of_files, len(results), total_time)

    return pd.DataFrame(results).sort_values(["Ticker", "Date", "Accession", "Source_Type", "Block_ID"]).reset_index(drop=True)


###################### CZESC - MAIN ######################
def main() -> None:

    df_sentiment = analyze_sentiment_directory(input_dir=INPUT_DIRECTORY,
                                               checkpoint_path=CHECKPOINT_FILE)

    if df_sentiment.empty:
        raise RuntimeError('błąd - analiza nie wygenerowała żadnych wyników sentymentu')

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df_sentiment.to_csv(OUTPUT_FILE, index=False)

    logger.info("Zapisano %d bloków sentymentu do %s", len(df_sentiment), OUTPUT_FILE)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format=("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))

    main()
