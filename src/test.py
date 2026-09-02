from pathlib import Path
import logging

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILE = PROJECT_ROOT / "data" / "processed" / "sentiment_filing_features.csv"

def main() -> None:
    df = pd.read_csv(FILE)
    columns = ["Ticker", "Filing_Date", "Accession", "Source_Types", "Filing_Block_Count",
               "EightK_Block_Count", "EX99_Block_Count", "Sentiment_Source"]
    logger.info("Największe filingi:\n%s", df.nlargest(10, "Filing_Block_Count")[columns].to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    main()
