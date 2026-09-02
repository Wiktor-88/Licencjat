# XAI dla TabNetu

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.compose import ColumnTransformer

from src.xai.common import (build_event_metadata, get_local_xai_dir,
    get_model_xai_dir, save_dataframe, save_json)


logger = logging.getLogger(__name__)


class TabNetPredictor:
    # Wrapper pozwala liczyć PFI na oryginalnych cechach
    def __init__(self, model: TabNetClassifier, preprocessor: ColumnTransformer):
        self.model = model
        self.preprocessor = preprocessor

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        values = self.preprocessor.transform(X).astype(np.float32)

        if not np.isfinite(values).all():
            raise ValueError("NaN lub Inf po preprocessingu TabNet")

        return values

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._transform(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return preprocessor.get_feature_names_out().astype(str).tolist()


def extract_native_importance(model: TabNetClassifier, 
                              feature_names: list[str]) -> pd.DataFrame:
    importance = np.asarray(model.feature_importances_, dtype=float)

    if len(importance) != len(feature_names):
        raise ValueError("Liczba feature_importances_ nie zgadza się z liczbą cech")

    result = pd.DataFrame({"Feature": feature_names, "Native_Importance": importance})

    return (result.sort_values("Native_Importance", ascending=False)
        .reset_index(drop=True).assign(Rank=lambda x: np.arange(1, len(x) + 1)))


def calculate_tabnet_masks(model: TabNetClassifier,
                           X_processed: np.ndarray,
                           feature_names: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    explain, masks = model.explain(X_processed.astype(np.float32), normalize=True)

    explain = np.asarray(explain, dtype=float)

    if explain.shape != X_processed.shape:
        raise ValueError(f"Niepoprawny kształt explain {explain.shape}, oczekiwano {X_processed.shape}")

    result = pd.DataFrame({"Feature": feature_names,
                           "Mean_Explain_Weight": explain.mean(axis=0),
                           "Std_Explain_Weight": explain.std(axis=0)})

    for step, mask in sorted(masks.items()):
        mask = np.asarray(mask, dtype=float)

        if mask.shape != X_processed.shape:
            raise ValueError(f"Niepoprawny kształt maski kroku {step}: {mask.shape}")

        result[f"Mean_Mask_Step_{step + 1}"] = mask.mean(axis=0)

    result = (result.sort_values("Mean_Explain_Weight", ascending=False)
        .reset_index(drop=True).assign(Rank=lambda x: np.arange(1, len(x) + 1)))

    return result, explain


def summarize_tabnet_importance(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    result = (df.groupby(["Variant", "Feature"], as_index=False).agg(
            Mean_Importance=(value_column, "mean"),
            Std_Importance_Across_Folds=(value_column, "std"),
            N_Folds=("Test_Year", "nunique")))

    result["Presence_Rate"] = result["N_Folds"] / 4

    return (result.sort_values(["Variant", "Mean_Importance"], ascending=[True, False])
        .reset_index(drop=True))


def build_local_mask_table(model: TabNetClassifier, 
                           X_processed: np.ndarray,
                           feature_names: list[str]) -> pd.DataFrame:
    
    explain, masks = model.explain(X_processed.astype(np.float32), normalize=True)

    explain = np.asarray(explain, dtype=float)

    if len(explain) != 1:
        raise ValueError("Local XAI wymaga jednej obserwacji")

    result = pd.DataFrame({"Feature": feature_names,
                           "Feature_Value_Model_Input": X_processed[0],
                           "Explain_Weight": explain[0]})

    for step, mask in sorted(masks.items()):
        result[f"Mask_Step_{step + 1}"] = np.asarray(mask, dtype=float)[0]

    result["Abs_Explain_Weight"] = np.abs(result["Explain_Weight"])

    return (result.sort_values("Abs_Explain_Weight", ascending=False)
        .reset_index(drop=True).assign(Rank=lambda x: np.arange(1, len(x) + 1)))


def plot_tabnet_importance(importance_df: pd.DataFrame,
                           output_file: Path,
                           value_column: str,
                           top_n: int = 15,
                           title: str = 'TabNet feature importance') -> None:
    plot_df = importance_df.nlargest(top_n, value_column).sort_values(value_column)
    

    fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * len(plot_df))))
    ax.barh(plot_df["Feature"], plot_df[value_column])
    ax.set_xlabel(value_column)
    ax.set_ylabel("Cecha")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Zapisano wykres TabNet: %s", output_file)


def plot_step_masks(mask_df: pd.DataFrame, output_file: Path, top_n: int = 15) -> None:
    step_columns = [col for col in mask_df.columns
                    if col.startswith("Mean_Mask_Step_")]

    plot_df = mask_df.nlargest(top_n, "Mean_Explain_Weight").set_index("Feature")
    

    values = plot_df[step_columns].to_numpy()

    fig, ax = plt.subplots(figsize=(9, max(5, 0.45 * len(plot_df))))
    image = ax.imshow(values, aspect="auto")

    ax.set_xticks(np.arange(len(step_columns)))
    ax.set_xticklabels(step_columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(plot_df.index)

    ax.set_title("Średnie maski TabNet według kroków")
    fig.colorbar(image, ax=ax, label="Średnia waga maski")

    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Zapisano maski TabNet: %s", output_file)


def run_global_tabnet_xai(model: TabNetClassifier,
                          X_processed: np.ndarray,
                          feature_names: list[str],
                          xai_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = get_model_xai_dir(xai_name)

    native = extract_native_importance(model, feature_names)

    masks, _ = calculate_tabnet_masks(model, X_processed, feature_names)

    save_dataframe(native, output_dir / "native_importance.csv")
    save_dataframe(masks, output_dir / "mask_importance.csv")

    plot_tabnet_importance(native,
                           output_dir / "native_importance.png",
                          "Native_Importance",
                           title="Natywna ważność cech TabNet")

    plot_tabnet_importance(masks,
                           output_dir / "mask_importance.png",
                           "Mean_Explain_Weight",
                           title="Średnia ważność masek TabNet na OOS")

    plot_step_masks(masks, output_dir / "step_masks.png")

    return native, masks


def run_local_tabnet_xai(model: TabNetClassifier,
                         preprocessor: ColumnTransformer,
                         X_row: pd.DataFrame,
                         event_row: pd.Series,
                         xai_name: str) -> pd.DataFrame:
    X_processed = preprocessor.transform(X_row).astype(np.float32)

    if not np.isfinite(X_processed).all():
        raise ValueError("NaN lub Inf w local XAI TabNet")

    feature_names = get_feature_names(preprocessor)

    local = build_local_mask_table(model, X_processed, feature_names)

    accession = str(event_row["Accession"])
    output_dir = get_local_xai_dir(xai_name, accession)

    save_dataframe(local, output_dir / "mask_values.csv")

    plot_tabnet_importance(local.rename(columns={"Explain_Weight": "Mean_Explain_Weight"}),
                           output_dir / "mask_importance.png",
                           "Mean_Explain_Weight",
                           title="Local TabNet feature mask")

    probability = float(model.predict_proba(X_processed)[0, 1])

    event = build_event_metadata(event_row)

    if "XAI_Example_Type" in event_row.index:
        event["XAI_Example_Type"] = str(event_row["XAI_Example_Type"])

    save_json(
        {"event": event,"prediction": {"Probability_Class_1": probability,
                                       "Prediction": int(probability >= 0.5)}},
        output_dir / "summary.json")

    return local