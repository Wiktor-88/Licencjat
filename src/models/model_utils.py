# Plik robiący powtarzalne rzeczy, czyli liczenie metryk,
# walidacja i przygotowanie dnaych

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score)


logger = logging.getLogger(__name__)

TARGET_RETURN_COLUMNS = {
    "Target_Abnormal_1D": "Abnormal_Event_Return_1D",
    "Target_Tradable_Abnormal_1D": "Tradable_Abnormal_Return_1D"}

# Liczenie metryk
def calculate_metrics(y_true, y_pred, y_prob=None) -> dict:
    metrics = {"Accuracy": accuracy_score(y_true, y_pred),
                "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
                "Precision": precision_score(y_true, y_pred, zero_division=0),
                "Recall": recall_score(y_true, y_pred, zero_division=0),
                "F1": f1_score(y_true, y_pred, zero_division=0),
                "ROC_AUC": np.nan,
                "Brier_Score": np.nan,
                "Log_Loss": np.nan}

    if y_prob is not None and pd.Series(y_true).nunique() == 2:
        metrics["ROC_AUC"] = roc_auc_score(y_true, y_prob)
        metrics["Brier_Score"] = brier_score_loss(y_true, y_prob)
        metrics["Log_Loss"] = log_loss(y_true, y_prob, labels=[0, 1])

    return metrics


def add_confusion_metrics(result: dict, y_true, y_pred) -> None:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    result.update({"TN": int(tn),
                    "FP": int(fp),
                    "FN": int(fn),
                    "TP": int(tp)})


def aggregate_model_events(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Łączy kilka filingów spółki prowadzących do tej samej decyzji sesyjnej."""
    rows = []
    consistency_columns = [target, "Tradable_Abnormal_Return_1D", "Feature_Cutoff_Session"]

    for (_, _), event_df in df.groupby(["Ticker", "Event_Session"], sort=True):
        for column in consistency_columns:
            if column in event_df.columns and event_df[column].nunique(dropna=False) > 1:
                raise ValueError(f"Niespójna kolumna {column} wewnątrz jednego Ticker + Event_Session")

        event_df = event_df.copy()
        event_df["_Acceptance_UTC"] = pd.to_datetime(
            event_df["Acceptance_DateTime_ET"], format="mixed", errors="raise", utc=True)
        event_df = event_df.sort_values(["_Acceptance_UTC", "Accession"])
        row = event_df.iloc[-1].copy()
        row["Grouped_Accessions"] = " | ".join(event_df["Accession"].astype(str))
        row["Event_Filing_Count"] = len(event_df)

        binary_columns = [column for column in event_df.columns if column.startswith("Has_")]
        for column in binary_columns:
            row[column] = int(pd.to_numeric(event_df[column], errors="coerce").fillna(0).max())

        weights = (pd.to_numeric(event_df["Sentiment_Total_Tokens"], errors="coerce")
            if "Sentiment_Total_Tokens" in event_df.columns else pd.Series(1.0, index=event_df.index))
        weights = weights.where(weights.gt(0), 1.0)
        if "Session_Mean_Net_Sentiment" in event_df.columns:
            row["Mean_Net_Sentiment"] = event_df["Session_Mean_Net_Sentiment"].iloc[0]
        else:
            row["Mean_Net_Sentiment"] = np.average(event_df["Mean_Net_Sentiment"], weights=weights)
        if "Sentiment_Total_Tokens" in event_df.columns:
            row["Sentiment_Total_Tokens"] = float(weights.sum())

        rows.append(row.drop(labels="_Acceptance_UTC"))

    result = pd.DataFrame(rows).reset_index(drop=True)
    if result.duplicated(["Ticker", "Event_Session"]).any():
        raise ValueError("Agregacja nie utworzyła unikalnych decyzji Ticker + Event_Session")
    return result


def prepare_model_dataset(df: pd.DataFrame, target: str) -> pd.DataFrame:

    df = df.copy()

    df["Event_Session"] = pd.to_datetime(df["Event_Session"], errors="raise")

    df = df[(df["Use_In_Primary_Model"] == 1) & df[target].notna()].copy()

    df[target] = df[target].astype(int)

    if not set(df[target].unique()).issubset({0, 1}):
        raise ValueError("Target musi przyjmować tylko wartości 0 i 1")

    duplicate_mask = df.duplicated(["Ticker", "Accession"], keep=False)

    if duplicate_mask.any():
        duplicates = df.loc[duplicate_mask, ["Ticker", "Accession"]]

        raise ValueError(f"Znaleziono zduplikowane filingi:\n"
                         f"{duplicates.to_string(index=False)}")

    df = aggregate_model_events(df, target)

    # Rozróżnia brak historii od tego, że momentum = 0
    history_count = pd.to_numeric(df["Sentiment_History_Count_3"], errors="coerce",).fillna(0)

    df["Has_Sentiment_History"] = (history_count > 0).astype(int)

    return df.sort_values(["Event_Session", "Ticker", "Accession"]).reset_index(drop=True)



def add_sentiment_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Abs_Sentiment"] = pd.to_numeric(df["Mean_Net_Sentiment"], errors="coerce").abs()

    df["Sentiment_x_Prior_Return_5D"] = (pd.to_numeric(df["Mean_Net_Sentiment"], errors="coerce")
        * pd.to_numeric(df["Stock_vs_QQQ_5D"], errors="coerce"))

    return df


# Sprawdzenie poprwności targetu
def validate_target(df: pd.DataFrame, target: str) -> None:
    return_column = TARGET_RETURN_COLUMNS.get(target)
    if return_column is None:
        raise ValueError(f"Brak przypisanej stopy zwrotu dla targetu: {target}")

    abnormal_return = pd.to_numeric(df[return_column], errors="coerce")

    if abnormal_return.isna().any():
        raise ValueError(f"Brakujące wartości {return_column}")

    expected_target = (abnormal_return > 0).astype(int)

    invalid = expected_target != df[target]

    if invalid.any():
        rows = df.loc[invalid, ["Ticker", "Accession", return_column, target]]

        raise ValueError("Target nie zgadza się z abnormal return:\n"
                        f"{rows.to_string(index=False)}")


# Wybieranie cech SEC
def select_sec_features(train_df: pd.DataFrame, candidates: list[str], min_count: int = 5) -> list[str]:

    selected = []

    for feature in candidates:
        count = int(train_df[feature].sum())

        if feature == "Has_EX99" or count >= min_count:
            selected.append(feature)

    return selected


# 
def log_repeated_events(df: pd.DataFrame) -> None:
    counts = df.groupby(["Ticker", "Event_Session"]).size()

    repeated = counts[counts > 1]

    logger.info("Pary Ticker + Event_Session z więcej niż jednym filingiem: %d",
                len(repeated))

    if not repeated.empty:
        logger.info("Największe powtórzenia:\n%s",
                    repeated.sort_values(ascending=False).head(10).to_string())


# Robienie podsumowania
def create_summary(results_df: pd.DataFrame, predictions_df: pd.DataFrame) -> pd.DataFrame:

    metric_columns = [
        "Accuracy",
        "Balanced_Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
        "Brier_Score",
        "Log_Loss",
    ]

    yearly_mean = results_df.groupby("Model")[metric_columns].mean().add_prefix("Mean_Yearly_").reset_index()
    

    pooled_rows = []

    for model_name, model_df in predictions_df.groupby("Model"):
        metrics = calculate_metrics(y_true=model_df["y_true"].astype(int),
                                    y_pred=model_df["y_pred"].astype(int),
                                    y_prob=model_df["y_prob"].astype(float))

        pooled_rows.append({
            "Model": model_name,
            **{f"Pooled_{metric}": value for metric, value in metrics.items()}})

    pooled = pd.DataFrame(pooled_rows)

    return (yearly_mean.merge(pooled, on="Model", how="left")
            .sort_values("Mean_Yearly_Balanced_Accuracy", ascending=False)
            .reset_index(drop=True))
