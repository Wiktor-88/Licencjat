# Wspólne Integrated Gradients dla LSTM i Transformera

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.model_config import TEST_YEARS
from src.xai.common import (build_event_metadata, get_local_xai_dir,
    get_model_xai_dir, save_dataframe, save_json)


logger = logging.getLogger(__name__)


def integrated_gradients(model,
                        sequence_x: np.ndarray,
                        static_x: np.ndarray,
                        device: torch.device,
                        n_steps: int = 32) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    model.eval()

    # cuDNN wymaga trybu train dla backward przez LSTM/RNN
    # Reszta modelu zostaje w eval, więc dropout nadal jest wyłączony
    for module in model.modules():
        if isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
            module.train()

    sequence = torch.tensor(sequence_x, dtype=torch.float32, device=device)
    static = torch.tensor(static_x, dtype=torch.float32, device=device)

    sequence_grads = torch.zeros_like(sequence)
    static_grads = torch.zeros_like(static)

    # Zero baseline ma sens po zastosowanym skalowaniu
    for alpha in (torch.arange(n_steps, device=device) + 0.5) / n_steps:
        seq_step = (sequence * alpha).detach().requires_grad_(True)
        stat_step = (static * alpha).detach().requires_grad_(True)

        logits = model(seq_step, stat_step)

        seq_grad, stat_grad = torch.autograd.grad(logits.sum(),(seq_step, stat_step))

        sequence_grads += seq_grad
        static_grads += stat_grad

    sequence_attr = sequence * sequence_grads / n_steps
    static_attr = static * static_grads / n_steps

    with torch.no_grad():
        logits = model(sequence, static)
        baseline_logits = model(torch.zeros_like(sequence), torch.zeros_like(static))

    reconstructed = (sequence_attr.flatten(1).sum(dim=1) + static_attr.sum(dim=1))

    difference = logits - baseline_logits
    errors = torch.abs(reconstructed - difference)

    check = pd.DataFrame({"Logit": logits.cpu().numpy(),
                           "Baseline_Logit": baseline_logits.cpu().numpy(),
                           "IG_Sum": reconstructed.cpu().numpy(),
                           "Logit_Difference": difference.cpu().numpy(),
                           "Completeness_Error": errors.cpu().numpy()})

    max_error = float(errors.max())

    if max_error > 0.05:
        logger.warning("Większy błąd completeness IG: %.6f. Można zwiększyć n_steps.",
                        max_error)

    model.eval()

    return (sequence_attr.detach().cpu().numpy(),
            static_attr.detach().cpu().numpy(),
            check)


def sequence_ig_table(attribution: np.ndarray, 
                      feature_names: list[str]) -> pd.DataFrame:
    rows = []

    for step in range(attribution.shape[1]):
        for feature_id, feature in enumerate(feature_names):
            values = attribution[:, step, feature_id]

            rows.append({"Sequence_Step": step + 1,
                         "Feature": feature,
                         "Mean_Abs_IG": float(np.mean(np.abs(values))),
                         "Mean_IG": float(np.mean(values)),
                         "Positive_Share": float(np.mean(values > 0)),
                         "Negative_Share": float(np.mean(values < 0))})

    return pd.DataFrame(rows)


def static_ig_table(attribution: np.ndarray,
                    feature_names: list[str]) -> pd.DataFrame:
    return (pd.DataFrame({"Feature": feature_names,
                          "Mean_Abs_IG": np.mean(np.abs(attribution), axis=0),
                          "Mean_IG": np.mean(attribution, axis=0),
                          "Positive_Share": np.mean(attribution > 0, axis=0),
                          "Negative_Share": np.mean(attribution < 0, axis=0)})
        .sort_values("Mean_Abs_IG", ascending=False).reset_index(drop=True))


def summarize_sequence_ig(df: pd.DataFrame) -> pd.DataFrame:
    result = (df.groupby(["Variant", "Sequence_Step", "Feature"], as_index=False).agg(
            Mean_Abs_IG=("Mean_Abs_IG", "mean"),
            Std_Abs_IG_Across_Folds=("Mean_Abs_IG", "std"),
            Mean_IG=("Mean_IG", "mean"),
            N_Folds=("Test_Year", "nunique")))

    result["Presence_Rate"] = result["N_Folds"] / len(TEST_YEARS)

    return result.sort_values(["Variant", "Mean_Abs_IG"], ascending=[True, False]).reset_index(drop=True)


def summarize_static_ig(df: pd.DataFrame) -> pd.DataFrame:
    result = (df.groupby(["Variant", "Feature"], as_index=False).agg(
            Mean_Abs_IG=("Mean_Abs_IG", "mean"),
            Std_Abs_IG_Across_Folds=("Mean_Abs_IG", "std"),
            Mean_IG=("Mean_IG", "mean"),
            N_Folds=("Test_Year", "nunique")))

    result["Presence_Rate"] = result["N_Folds"] / len(TEST_YEARS)

    return result.sort_values(["Variant", "Mean_Abs_IG"], ascending=[True, False]).reset_index(drop=True)


def plot_sequence_heatmap(df: pd.DataFrame,
                          output_file: Path,
                          value_column: str = "Mean_IG",
                          title: str = "Integrated Gradients – kierunek i siła wpływu",
                          colorbar_label: str = "Średnia wartość IG") -> None:
    matrix = (df.pivot_table(index="Feature",
                            columns="Sequence_Step",
                            values=value_column,
                            aggfunc="mean"))

    fig, ax = plt.subplots(figsize=(14, max(5, 0.55 * len(matrix))))

    values = matrix.to_numpy(dtype=float)
    limit = float(np.nanmax(np.abs(values)))
    limit = limit if np.isfinite(limit) and limit > 0 else 1.0

    image = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    ax.set_xlabel("Krok sekwencji")
    ax.set_ylabel("Cecha")
    ax.set_title(title)

    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Zapisano heatmapę IG: %s", output_file)


def plot_static_ig(df: pd.DataFrame, output_file: Path, top_n: int = 15) -> None:
    plot_df = (df.nlargest(top_n, "Mean_Abs_IG").sort_values("Mean_Abs_IG"))

    fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * len(plot_df))))

    ax.barh(plot_df["Feature"], plot_df["Mean_Abs_IG"])

    ax.set_xlabel("Mean |IG|")
    ax.set_ylabel("Cecha statyczna")
    ax.set_title("Integrated Gradients – cechy statyczne")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_global_sequence_ig(model,
                           sequence_x: np.ndarray,
                           static_x: np.ndarray,
                           sequence_features: list[str],
                           static_features: list[str],
                           device: torch.device,
                           xai_name: str,
                           n_steps: int = 32) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = get_model_xai_dir(xai_name)

    sequence_attr, static_attr, check = integrated_gradients(model,
                                                            sequence_x,
                                                            static_x,
                                                            device,
                                                            n_steps)

    sequence_df = sequence_ig_table(sequence_attr, sequence_features)

    static_df = static_ig_table(static_attr, static_features)

    save_dataframe(sequence_df, output_dir / "sequence_ig.csv")
    save_dataframe(static_df, output_dir / "static_ig.csv")
    save_dataframe(check, output_dir / "ig_completeness.csv")

    plot_sequence_heatmap(sequence_df, output_dir / "sequence_ig_heatmap.png")

    plot_static_ig(static_df, output_dir / "static_ig.png")

    return sequence_df, static_df


def run_local_sequence_ig(model,
                          sequence_x: np.ndarray,
                          static_x: np.ndarray,
                          sequence_features: list[str],
                          static_features: list[str],
                          event_row: pd.Series,
                          device: torch.device,
                          xai_name: str,
                          n_steps: int = 32) -> None:
    sequence_attr, static_attr, check = integrated_gradients(model,
                                                            sequence_x,
                                                            static_x,
                                                            device,
                                                            n_steps)

    sequence_rows = []

    for step in range(sequence_attr.shape[1]):
        for feature_id, feature in enumerate(sequence_features):
            value = float(sequence_attr[0, step, feature_id])

            sequence_rows.append({"Sequence_Step": step + 1,
                                  "Feature": feature,
                                  "Model_Input_Value": float(sequence_x[0, step, feature_id]),
                                  "IG_Value": value,
                                  "Abs_IG_Value": abs(value)})

    sequence_df = pd.DataFrame(sequence_rows)

    static_df = pd.DataFrame({"Feature": static_features,
                              "Model_Input_Value": static_x[0],
                              "IG_Value": static_attr[0],
                              "Abs_IG_Value": np.abs(static_attr[0])}).sort_values("Abs_IG_Value", ascending=False)

    accession = str(event_row["Accession"])
    output_dir = get_local_xai_dir(xai_name, accession)

    save_dataframe(sequence_df, output_dir / "sequence_ig.csv")
    save_dataframe(static_df, output_dir / "static_ig.csv")
    save_dataframe(check, output_dir / "ig_completeness.csv")

    local_plot = sequence_df.rename(columns={"IG_Value": "Mean_IG"})

    plot_sequence_heatmap(local_plot,
                          output_dir / "sequence_ig_heatmap.png",
                          value_column="Mean_IG",
                          title="Lokalne Integrated Gradients – kierunek i siła wpływu",
                          colorbar_label="Wartość IG")

    event = build_event_metadata(event_row)

    if "XAI_Example_Type" in event_row.index:
        event["XAI_Example_Type"] = str( event_row["XAI_Example_Type"])

    save_json(
        {"event": event,
        "prediction": {"Logit": float(check.iloc[0]["Logit"]),
                       "Probability_Class_1": float(torch.sigmoid(torch.tensor(check.iloc[0]["Logit"])).item()),
                       "Completeness_Error": float(check.iloc[0]["Completeness_Error"])}},
        output_dir / "summary.json")
