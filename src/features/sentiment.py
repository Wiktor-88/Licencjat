import os
import glob
import time
import re

import pandas as pd
import nltk
import torch

from nltk.tokenize import sent_tokenize
from transformers import (
    pipeline,
    BertTokenizer,
    BertForSequenceClassification,
)


# ============================================================
# KONFIGURACJA
# ============================================================

MODEL_NAME = "yiyanghkust/finbert-tone"

# FinBERT/BERT ma limit 512 tokenów.
# Używamy 450, żeby zostawić bezpieczny margines.
MAX_TOKENS_PER_BLOCK = 450

# Liczba zdań przenoszonych z poprzedniego bloku.
SENTENCE_OVERLAP = 2

# ZMIANA:
# FinBERT będzie analizował do 16 bloków jednocześnie na GPU.
# Przy Twojej RTX 5070 Ti później możesz spróbować 32.
BATCH_SIZE = 32

# Co ile plików zapisujemy częściowe wyniki.
CHECKPOINT_EVERY = 50


# ============================================================
# GPU / CPU
# ============================================================

# ZMIANA:
# Nie ustawiamy device=0 na sztywno.
# Jeśli CUDA działa -> GPU.
# Jeśli nie -> automatycznie CPU.
DEVICE = 0 if torch.cuda.is_available() else -1

if DEVICE == 0:
    print(f"FinBERT na GPU: {torch.cuda.get_device_name(0)}")
else:
    print("FinBERT na CPU")

def ensure_nltk_resource(
    resource_path: str,
    download_name: str,
) -> None:
    """
    Sprawdza, czy wymagany zasób NLTK jest dostępny lokalnie.
    Pobiera go tylko wtedy, gdy go brakuje.
    """

    try:
        nltk.data.find(resource_path)

    except LookupError:
        print(
            f"Brakuje zasobu NLTK: {download_name}. "
            "Próba pobrania..."
        )

        success = nltk.download(
            download_name,
            quiet=False,
        )

        if not success:
            raise RuntimeError(
                f"Nie udało się pobrać zasobu NLTK: "
                f"{download_name}"
            )


ensure_nltk_resource(
    "tokenizers/punkt",
    "punkt",
)

ensure_nltk_resource(
    "tokenizers/punkt_tab/english",
    "punkt_tab",
)

ensure_nltk_resource(
    "tokenizers/punkt",
    "punkt",
)

ensure_nltk_resource(
    "tokenizers/punkt_tab/english",
    "punkt_tab",
)


def parse_file_metadata(
    filename: str,
) -> tuple[str, str, str, str]:
    """
    Wyciąga:
    - ticker
    - datę filing
    - accession number
    - typ dokumentu

    Przykłady:

    AAPL_2026-01-29_0000320193-26-000005_8-K.txt

    AAPL_2026-01-29_0000320193-26-000005_EX-99-1.txt
    """

    filename_without_extension = os.path.splitext(
        filename
    )[0]

    parts = filename_without_extension.split(
        "_",
        3,
    )

    if len(parts) != 4:
        raise ValueError(
            f"Nieprawidłowy format nazwy pliku: "
            f"{filename}"
        )

    ticker = parts[0]
    filing_date = parts[1]
    accession_number = parts[2]
    source_type = parts[3]

    # process_pipeline zapisuje np.:
    # EX-99.1 -> EX-99-1
    #
    # Tutaj przywracamy oryginalną nazwę SEC.
    if source_type.startswith("EX-99-"):
        source_type = source_type.replace(
            "EX-99-",
            "EX-99.",
            1,
        )

    return (
        ticker,
        filing_date,
        accession_number,
        source_type,
    )



def split_into_8k_items(
    text: str,
) -> list[tuple[str, str]]:
    """
    Dzieli oczyszczony raport 8-K na osobne sekcje ITEM.

    Przykład:

        Item 2.02. Results of Operations...
        ...
        Item 8.01. Other Events...
        ...

    zostanie zamienione na:

        [
            ("2.02", "Item 2.02 ..."),
            ("8.01", "Item 8.01 ..."),
        ]

    Dzięki temu każdy późniejszy blok FinBERT-a
    można jednoznacznie powiązać z sekcją raportu.
    """

    # Szukamy wszystkich nagłówków typu:
    #
    # Item 1.01
    # ITEM 2.02.
    # item 5.02
    item_matches = list(
        re.finditer(
            r"\bITEM\s+(\d+\.\d+)\.?",
            text,
            flags=re.IGNORECASE,
        )
    )

    # Teoretycznie clean_text.py powinien już zapewnić
    # przynajmniej jeden ITEM.
    #
    # Zostawiamy jednak fallback dla bezpieczeństwa.
    if not item_matches:
        return [
            ("UNKNOWN", text)
        ]

    item_sections = []

    for index, match in enumerate(
        item_matches
    ):
        item_number = match.group(1)

        section_start = match.start()

        # Koniec sekcji = początek następnego ITEM.
        if index + 1 < len(item_matches):
            section_end = (
                item_matches[index + 1].start()
            )
        else:
            section_end = len(text)

        section_text = text[
            section_start:section_end
        ].strip()

        if section_text:
            item_sections.append(
                (
                    item_number,
                    section_text,
                )
            )

    return item_sections



# ============================================================
# TWORZENIE BLOKÓW
# ============================================================

def create_sentence_blocks(
    text: str,
    tokenizer,
    max_tokens: int = MAX_TOKENS_PER_BLOCK,
    overlap_sentences: int = SENTENCE_OVERLAP,
) -> list[str]:
    """
    Dzieli dokument na bloki składające się z pełnych zdań.

    Każdy blok ma maksymalnie około max_tokens tokenów FinBERT-a.

    Między kolejnymi blokami zachowywany jest overlap kilku zdań.

    ZMIANA:
    Poprzednia wersja używała:
        while i < len(sentences)

    i w pewnych sytuacjach indeks i nie był zwiększany.
    Mogło to prowadzić do nieskończonej pętli.

    Tutaj używamy zwykłego:
        for sentence in sentences

    dzięki czemu nie ma możliwości utknięcia na jednym zdaniu.
    """

    sentences = sent_tokenize(
        text,
        language="english",
    )

    blocks = []

    current_block = []
    current_lengths = []
    current_token_count = 0

    for sentence in sentences:

        # Tokenizujemy zdanie tokenizerem FinBERT-a,
        # a NIE tokenizerem NLTK.
        sentence_token_ids = tokenizer.encode(
            sentence,
            add_special_tokens=False,
        )

        sentence_length = len(sentence_token_ids)

        # ====================================================
        # PRZYPADEK 1:
        # pojedyncze zdanie samo ma > 450 tokenów
        # ====================================================

        if sentence_length > max_tokens:

            # Najpierw zapisujemy aktualnie budowany blok.
            if current_block:
                blocks.append(
                    " ".join(current_block)
                )

                current_block = []
                current_lengths = []
                current_token_count = 0

            # ZMIANA:
            # Bardzo długiego zdania nie wysyłamy w całości.
            # Dzielimy je bezpośrednio według tokenów.
            for start in range(
                0,
                sentence_length,
                max_tokens,
            ):
                token_fragment = sentence_token_ids[
                    start:start + max_tokens
                ]

                fragment_text = tokenizer.decode(
                    token_fragment,
                    skip_special_tokens=True,
                )

                if fragment_text.strip():
                    blocks.append(fragment_text)

            continue

        # ====================================================
        # PRZYPADEK 2:
        # zdanie mieści się w aktualnym bloku
        # ====================================================

        if (
            current_token_count + sentence_length
            <= max_tokens
        ):
            current_block.append(sentence)
            current_lengths.append(sentence_length)

            current_token_count += sentence_length

            continue

        # ====================================================
        # PRZYPADEK 3:
        # dodanie zdania przekroczyłoby limit
        # ====================================================

        if current_block:
            blocks.append(
                " ".join(current_block)
            )

        # ZMIANA:
        # Zachowujemy ostatnie np. 2 zdania jako overlap.
        if overlap_sentences > 0:
            overlap = current_block[
                -overlap_sentences:
            ]

            overlap_lengths = current_lengths[
                -overlap_sentences:
            ]
        else:
            overlap = []
            overlap_lengths = []

        overlap_token_count = sum(
            overlap_lengths
        )

        # ZMIANA:
        # Jeżeli overlap + nowe zdanie nadal przekracza 450,
        # zmniejszamy overlap.
        #
        # To właśnie zabezpiecza przypadek, który wcześniej
        # mógł doprowadzić do nieskończonej pętli.
        while (
            overlap
            and overlap_token_count + sentence_length
            > max_tokens
        ):
            removed_length = overlap_lengths.pop(0)
            overlap.pop(0)

            overlap_token_count -= removed_length

        # Nowe zdanie ZAWSZE zostaje dodane.
        # Nie cofamy żadnego indeksu.
        current_block = overlap + [sentence]

        current_lengths = (
            overlap_lengths
            + [sentence_length]
        )

        current_token_count = (
            overlap_token_count
            + sentence_length
        )

    # ========================================================
    # OSTATNI BLOK
    # ========================================================

    if current_block:
        blocks.append(
            " ".join(current_block)
        )

    return blocks


# ============================================================
# ANALIZA SENTYMENTU
# ============================================================

def analyze_sentiment_sentence_block(
    input_dir: str,
    checkpoint_path: str | None = None,
    limit_files: int | None = None,
    test_files_per_ticker: int | None = None,
) -> pd.DataFrame:

    # ========================================================
    # ŁADOWANIE MODELU
    # ========================================================


    tokenizer = BertTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
    )

    # ZMIANA:
    # Model i tokenizer są ładowane tylko RAZ,
    # przed rozpoczęciem analizy plików.
    sentiment_pipeline = pipeline(
        task="sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        top_k=None,
    )

    print("Model załadowany.\n")

    ##################################################
    # TESTY POTEM USNAC
    ##################################################

    test_sentences = [
    "The company reported record revenue and strong profit growth.",
    "The company reported significant losses and warned about declining demand.",
    "The company filed a Form 8-K with the Securities and Exchange Commission.",
    ]

    test_predictions = sentiment_pipeline(
        test_sentences,
        top_k=None,
    )

    print("\n--- TEST FINBERT ---")

    for sentence, prediction in zip(
        test_sentences,
        test_predictions,
    ):
        print(f"\nTekst: {sentence}")

        for item in prediction:
            print(
                f"{item['label']}: "
                f"{item['score']:.4f}"
            )




    results = []

    # ========================================================
    # WYSZUKIWANIE PLIKÓW
    # ========================================================

    # sorted() daje powtarzalną kolejność.
    file_paths = sorted(
        glob.glob(
            os.path.join(
                input_dir,
                "*.txt",
            )
        )
    )

    # ========================================================
    # TRYB TESTOWY:
    # określona liczba raportów dla każdego tickera
    # ========================================================

    if test_files_per_ticker is not None:

        files_by_ticker = {}

        for file_path in file_paths:

            filename = os.path.basename(
                file_path
            )

            try:
                ticker, _, _, _ = parse_file_metadata(
                    filename
                )

            except ValueError:
                continue

            if ticker not in files_by_ticker:
                files_by_ticker[ticker] = []

            files_by_ticker[ticker].append(
                file_path
            )

        selected_files = []

        for ticker in sorted(files_by_ticker):

            ticker_files = files_by_ticker[
                ticker
            ]

            selected_files.extend(
                ticker_files[
                    :test_files_per_ticker
                ]
            )

        file_paths = selected_files

    # Klasyczny limit globalny zostawiamy
    # jako dodatkową możliwość.
    elif limit_files is not None:

        file_paths = file_paths[
            :limit_files
        ]



    number_of_files = len(file_paths)

    print(
        f"Liczba znalezionych plików: "
        f"{number_of_files}\n"
    )

    if number_of_files == 0:
        print(
            "Nie znaleziono żadnych plików .txt."
        )

        return pd.DataFrame()

    # ========================================================
    # ANALIZA PLIKÓW
    # ========================================================

    total_start_time = time.perf_counter()

    for file_index, file_path in enumerate(
        file_paths,
        start=1,
    ):

        file_start_time = time.perf_counter()

        filename = os.path.basename(
            file_path
        )

        try:
            (ticker, filing_date, accession_number, source_type) = parse_file_metadata(
                filename
            )
        except ValueError as error:
            print(
                f"  Błąd parsowania metadanych: {error}"
            )
            continue

        # ====================================================
        # WCZYTANIE TEKSTU
        # ====================================================

        try:
            with open(
                file_path,
                "r",
                encoding="utf-8",
            ) as file:
                text = file.read()

        except Exception as error:
            print(
                f"  Błąd odczytu pliku: {error}"
            )
            continue

        # Pomijamy puste pliki.
        if not text.strip():
            print("  Pusty plik - pominięto.")
            continue

        # ====================================================
        # CHUNKING
        # ====================================================

        chunk_start = time.perf_counter()

        # ====================================================
        # PODZIAŁ LOGICZNY DOKUMENTU
        # ====================================================

        if source_type == "8-K":

            # Główny formularz 8-K ma strukturę:
            #
            # Item 2.02
            # Item 5.02
            # Item 8.01
            # Item 9.01
            #
            # Dlatego dzielimy go według Item_Number.
            item_sections = split_into_8k_items(
                text
            )

        else:

            # EX-99.1 / EX-99.2 to np.:
            #
            # earnings release
            # press release
            # CFO commentary
            #
            # Nie posiadają struktury ITEM formularza 8-K.
            #
            # Cały exhibit traktujemy jako jedną logiczną
            # sekcję, którą później dzielimy na bloki.
            item_sections = [
                ("N/A", text)
            ]

        blocks_with_metadata = []

        global_block_id = 0

        for item_number, item_text in item_sections:

            # Każdy ITEM dzielimy osobno na bloki FinBERT-a
            item_blocks = create_sentence_blocks(
                text=item_text,
                tokenizer=tokenizer,
                max_tokens=MAX_TOKENS_PER_BLOCK,
                overlap_sentences=SENTENCE_OVERLAP,
            )

            for item_block_id, block_text in enumerate(item_blocks):

                blocks_with_metadata.append({
                    "Block_ID": global_block_id,
                    "Item_Number": item_number,
                    "Item_Block_ID": item_block_id,
                    "Text": block_text,
                })

                global_block_id += 1


        # Lista samych tekstów przekazywana do FinBERT-a
        blocks = [
            block["Text"]
            for block in blocks_with_metadata
        ]


        chunk_time = (
            time.perf_counter()
            - chunk_start
        )

        print(
            f"  Bloki: {len(blocks)} "
            f"| chunking: {chunk_time:.2f}s"
        )

        if not blocks:
            print(
                "  Nie utworzono żadnych bloków."
            )
            continue

        # ====================================================
        # FINBERT
        # ====================================================

        finbert_start = time.perf_counter()

        try:

            # ZMIANA KLUCZOWA:
            #
            # STARA WERSJA:
            #
            # for block in blocks:
            #     sentiment_pipeline(block)
            #
            # czyli każda inferencja osobno.
            #
            #
            # NOWA WERSJA:
            #
            # przekazujemy CAŁĄ LISTĘ.
            #
            # pipeline sam tworzy batch'e po 16.
            all_predictions = sentiment_pipeline(
                blocks,
                truncation=True,
                max_length=512,
                batch_size=BATCH_SIZE,
            )

        except RuntimeError as error:

            # Osobna informacja dla typowego błędu GPU.
            if "out of memory" in str(error).lower():

                print(
                    "  CUDA OUT OF MEMORY!"
                )

                print(
                    "  Zmniejsz BATCH_SIZE, "
                    "np. z 16 do 8."
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            else:
                print(
                    f"  RuntimeError: {error}"
                )

            continue

        except Exception as error:
            print(
                f"  Błąd FinBERT: {error}"
            )
            continue

        finbert_time = (
            time.perf_counter()
            - finbert_start
        )

        # ====================================================
        # ZAPIS WYNIKÓW BLOKÓW
        # ====================================================

        for block_metadata, predictions in zip(
            blocks_with_metadata,
            all_predictions,
        ):

            try:

                scores = {
                    item["label"].lower():
                        float(item["score"])
                    for item in predictions
                }

                prob_positive = scores["positive"]
                prob_negative = scores["negative"]
                prob_neutral = scores["neutral"]

                # Wynik od około -1 do +1
                net_sentiment = (
                    prob_positive
                    - prob_negative
                )

                # Etykieta klasy o najwyższym prawdopodobieństwie
                predicted_label = max(
                    scores,
                    key=scores.get,
                )

                # Pewność modelu dla wybranej klasy
                confidence = scores[
                    predicted_label
                ]

                results.append({

                    # ==========================================
                    # METADANE RAPORTU
                    # ==========================================

                    "Ticker": ticker,
                    "Date": filing_date,
                    "Accession": accession_number,
                    "Source_Type": source_type,
                    "File": filename,

                    # ==========================================
                    # STRUKTURA RAPORTU
                    # ==========================================

                    "Item_Number":
                        block_metadata["Item_Number"],

                    "Block_ID":
                        block_metadata["Block_ID"],

                    "Item_Block_ID":
                        block_metadata["Item_Block_ID"],

                    # ==========================================
                    # TEKST
                    # ==========================================

                    "Text_Snippet":
                        block_metadata["Text"],

                    # ==========================================
                    # FINBERT
                    # ==========================================

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
                                    })

            except Exception as error:
                print(
                    f"  Błąd zapisu bloku "
                    f"{block_metadata['Block_ID']}: "
                    f"{error}"
                )

        # ====================================================
        # CZAS
        # ====================================================

        file_time = (
            time.perf_counter()
            - file_start_time
        )

        blocks_per_second = (
            len(blocks) / finbert_time
            if finbert_time > 0
            else 0
        )

        print(
            f"  FinBERT: {finbert_time:.2f}s "
            f"| {blocks_per_second:.2f} bloków/s"
        )

        print(
            f"  Cały plik: {file_time:.2f}s"
        )

        # ====================================================
        # CHECKPOINT
        # ====================================================

        # ZMIANA:
        # Co np. 50 plików zapisujemy wyniki.
        #
        # Jeżeli skrypt padnie po kilku tysiącach plików,
        # nie tracisz całej pracy.
        if (
            checkpoint_path is not None
            and file_index % CHECKPOINT_EVERY == 0
        ):
            checkpoint_df = pd.DataFrame(
                results
            )

            checkpoint_df.to_csv(
                checkpoint_path,
                index=False,
            )

            print(
                f"  CHECKPOINT zapisany: "
                f"{checkpoint_path}"
            )

        print()

    # ========================================================
    # KONIEC
    # ========================================================

    total_time = (
        time.perf_counter()
        - total_start_time
    )

    print(
        "=" * 60
    )

    print(
        f"Zakończono analizę."
    )

    print(
        f"Łączny czas: "
        f"{total_time:.2f}s"
    )

    print(
        f"Liczba wynikowych bloków: "
        f"{len(results)}"
    )

    print(
        "=" * 60
    )

    return pd.DataFrame(
        results
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    INPUT_DIRECTORY = os.path.abspath(
        os.path.join(
            "data",
            "processed",
            "cleaned_texts",
        )
    )

    OUTPUT_DIRECTORY = os.path.abspath(
        os.path.join(
            "data",
            "processed",
        )
    )

    # Tworzymy katalog, jeśli nie istnieje.
    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True,
    )

    OUTPUT_FILE = os.path.join(
        OUTPUT_DIRECTORY,
        "sentiment_blocks.csv",
    )

    CHECKPOINT_FILE = os.path.join(
        OUTPUT_DIRECTORY,
        "sentiment_checkpoint.csv",
    )

    # ========================================================
    # TEST
    # ========================================================


    df_sentiment = (
        analyze_sentiment_sentence_block(
            input_dir=INPUT_DIRECTORY,
            checkpoint_path=CHECKPOINT_FILE,
            test_files_per_ticker=None,
            limit_files=None
        )
    )

    # ========================================================
    # PODGLĄD
    # ========================================================

    print(
        "\n--- Wygenerowana Ramka Danych ---"
    )

    columns_to_show = [
        "Ticker",
        "Date",
        "Source_Type",
        "Item_Number",
        "Block_ID",
        "Item_Block_ID",
        "Predicted_Label",
        "Confidence",
        "Prob_Positive",
        "Prob_Negative",
        "Prob_Neutral",
        "Net_Sentiment",
    ]

    print(
        df_sentiment[
            columns_to_show
        ].to_string(
            index=False
        )
    )

    # PODGLAD
    print(
        "\n--- ROZKŁAD KLAS ---"
    )

    print(
        df_sentiment[
            "Predicted_Label"
        ].value_counts()
    )


    print(
        "\n--- ROZKŁAD KLAS DLA SPÓŁEK ---"
    )

    print(
        pd.crosstab(
            df_sentiment["Ticker"],
            df_sentiment["Predicted_Label"],
        )
    )   



    #### SREDNIE PRAWDOPODOBIENSTWA
    print(
        "\n--- ŚREDNIE PRAWDOPODOBIEŃSTWA ---"
    )

    print(
        df_sentiment.groupby("Ticker")[
            [
                "Prob_Positive",
                "Prob_Negative",
                "Prob_Neutral",
            ]
        ].mean()
    )

    # BLKOKI NIE NEUTRLANE
    non_neutral = df_sentiment[
        df_sentiment["Predicted_Label"]
        != "Neutral"
    ]

    print(
        "\n--- BLOKI NIE-NEUTRALNE ---\n"
    )

    for _, row in non_neutral.iterrows():

        print(
            f"{row['Ticker']} | "
            f"{row['Date']} | "
            f"Item {row['Item_Number']}"
        )

        print(
            f"Label: {row['Predicted_Label']} "
            f"| Confidence: {row['Confidence']:.4f}"
        )

        print(
            row["Text_Snippet"][:500]
        )

        print("-" * 80)

        print(
        "\n--- ROZKŁAD KLAS WG SOURCE TYPE ---"
    )

    print(
        pd.crosstab(
            df_sentiment["Source_Type"],
            df_sentiment["Predicted_Label"],
        )
    )

    print(
        "\n--- ŚREDNIE PRAWDOPODOBIEŃSTWA WG SOURCE TYPE ---"
    )

    print(
        df_sentiment.groupby(
            "Source_Type"
        )[
            [
                "Prob_Positive",
                "Prob_Negative",
                "Prob_Neutral",
            ]
        ].mean()
    )

    
    

    # ========================================================
    # FINALNY ZAPIS
    # ========================================================

    df_sentiment.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nWyniki zapisano do:\n"
        f"{OUTPUT_FILE}"
    )