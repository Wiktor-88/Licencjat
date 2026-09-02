# XAI dla drzewa decyzyjnego

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree

from src.xai.common import (build_event_metadata, get_local_xai_dir,
    get_model_xai_dir, save_dataframe, save_json)


logger = logging.getLogger(__name__)

TreeModel = DecisionTreeClassifier | Pipeline


def get_tree_estimator(model: TreeModel) -> DecisionTreeClassifier:
    if isinstance(model, DecisionTreeClassifier):
        estimator = model
    elif isinstance(model, Pipeline):
        estimator = model.steps[-1][1]
    else:
        raise TypeError("Model musi być drzewm decyzyjnym lub Pipelinem")

    if not isinstance(estimator, DecisionTreeClassifier):
        raise TypeError("Końcowym estymatorem musi być DecisionTreeClassifier")
    if not hasattr(estimator, "tree_"):
        raise ValueError("DecisionTreeClassifier nie jest wytrenowany")
    if not np.array_equal(estimator.classes_, [0, 1]):
        raise ValueError('Inne klasy niż [0, 1]')

    return estimator


def transform_model_input(model: TreeModel, X: pd.DataFrame) -> np.ndarray:
    if not isinstance(X, pd.DataFrame) or X.empty:
        raise ValueError("X musi być niepustym DataFrame")

    transformed = (model[:-1].transform(X)
                   if isinstance(model, Pipeline) and len(model.steps) > 1
                   else X.to_numpy())

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(transformed, dtype=float)

    if transformed.ndim != 2 or not np.isfinite(transformed).all():
        raise ValueError("Niepoprawne dane po preprocessingu")

    return transformed


def resolve_feature_names(model: TreeModel,
                          X: pd.DataFrame,
                          transformed_X: np.ndarray,
                          feature_names: list[str] | None = None) -> list[str]:
    if feature_names is not None:
        names = [str(name) for name in feature_names]

    elif isinstance(model, Pipeline) and len(model.steps) > 1:
        try:
            names = model[:-1].get_feature_names_out().astype(str).tolist()
            
        except (AttributeError, ValueError):
            names = X.columns.astype(str).tolist()

    else:
        names = X.columns.astype(str).tolist()

    if len(names) != transformed_X.shape[1]:
        raise ValueError(f"Liczba nazw cech ({len(names)}) nie odpowiada "
                         f"liczbie cech modelu ({transformed_X.shape[1]})")

    return names


def extract_feature_importance(model: TreeModel,
                               X_reference: pd.DataFrame,
                               feature_names: list[str] | None = None ) -> pd.DataFrame:
    estimator = get_tree_estimator(model)
    transformed_X = transform_model_input(model, X_reference)

    names = resolve_feature_names(model,
                                  X_reference,
                                  transformed_X,
                                  feature_names)

    importance = estimator.feature_importances_.astype(float)

    result = pd.DataFrame({"Feature": names, "Impurity_Importance": importance})

    return (result.sort_values("Impurity_Importance", ascending=False)
            .reset_index(drop=True).assign(Rank=lambda x: np.arange(1, len(x) + 1)))


def extract_tree_structure(model: TreeModel,
                           X_reference: pd.DataFrame,
                           feature_names: list[str] | None = None) -> pd.DataFrame:
    estimator = get_tree_estimator(model)
    transformed_X = transform_model_input(model, X_reference)

    names = resolve_feature_names(model,
                                  X_reference,
                                  transformed_X,
                                  feature_names)

    tree = estimator.tree_
    rows = []

    for node_id in range(tree.node_count):
        feature_id = tree.feature[node_id]
        is_leaf = feature_id < 0

        rows.append({"Node": node_id,
                     "Is_Leaf": is_leaf,
                     "Feature": None if is_leaf else names[feature_id],
                     "Threshold": None if is_leaf else float(tree.threshold[node_id]),
                     "Left_Child": None if is_leaf else int(tree.children_left[node_id]),
                     "Right_Child": None if is_leaf else int(tree.children_right[node_id]),
                     "Samples": int(tree.n_node_samples[node_id]),
                     "Impurity": float(tree.impurity[node_id]),
                     "Class_0_Count": float(tree.value[node_id][0][0]),
                     "Class_1_Count": float(tree.value[node_id][0][1])})

    return pd.DataFrame(rows)


def calculate_decision_path(model: TreeModel,
                            X_row: pd.DataFrame,
                            feature_names: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    if len(X_row) != 1:
        raise ValueError("Lokalne XAI wymaga dokładnie jednej obserwacji")

    estimator = get_tree_estimator(model)
    transformed_X = transform_model_input(model, X_row)

    names = resolve_feature_names(model, X_row, transformed_X, feature_names)

    values = transformed_X[0]
    tree = estimator.tree_
    node_indicator = estimator.decision_path(transformed_X)
    leaf_id = int(estimator.apply(transformed_X)[0])

    node_ids = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

    rows = []

    for node_id in node_ids:
        if node_id == leaf_id:
            rows.append({"Node": int(node_id),
                         "Feature": None,
                         "Model_Input_Value": None,
                         "Threshold": None,
                         "Decision": "leaf",
                         "Next_Node": None})
            continue

        feature_id = tree.feature[node_id]
        value = float(values[feature_id])
        threshold = float(tree.threshold[node_id])

        if value <= threshold:
            decision = "<="
            next_node = int(tree.children_left[node_id])
        else:
            decision = ">"
            next_node = int(tree.children_right[node_id])

        rows.append({"Node": int(node_id),
                     "Feature": names[feature_id],
                     "Model_Input_Value": value,
                     "Threshold": threshold,
                     "Decision": decision,
                     "Next_Node": next_node})

    probability = float(model.predict_proba(X_row)[0, 1])
    prediction = int(model.predict(X_row)[0])

    summary = {"Leaf_Node": leaf_id,
               "Probability_Class_1": probability,
               "Prediction": prediction,
               "Path_Length": len(node_ids) - 1}

    return pd.DataFrame(rows), summary


# Wykres drzewa
def plot_tree_structure(model: TreeModel,
                        X_reference: pd.DataFrame,
                        output_file: Path,
                        feature_names: list[str] | None = None) -> None:
    
    estimator = get_tree_estimator(model)
    transformed_X = transform_model_input(model, X_reference)

    names = resolve_feature_names(model,
                                  X_reference,
                                  transformed_X,
                                  feature_names)

    fig, ax = plt.subplots(figsize=(22, 12))

    plot_tree(estimator,
              feature_names=names,
              class_names=["0", "1"],
              filled=True,
              rounded=True,
              impurity=True,
              proportion=True,
              ax=ax)

    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Zapisano strukturę drzewa: %s", output_file)



# Wykres ważności
def plot_feature_importance(importance_df: pd.DataFrame,
                            output_file: Path,
                            top_n: int = 15) -> None:
    plot_df = (importance_df.nlargest(top_n, "Impurity_Importance")
                .sort_values("Impurity_Importance"))

    fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * len(plot_df))))

    ax.barh(plot_df["Feature"], plot_df["Impurity_Importance"])

    ax.set_xlabel("Impurity-based feature importance")
    ax.set_ylabel("Cecha")
    ax.set_title("Drzewo decyzyjne – ważność cech")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Zapisano wykres ważności cech: %s", output_file)


def run_global_tree_xai(model: TreeModel,
                        X_reference: pd.DataFrame,
                        xai_name: str,
                        feature_names: list[str] | None = None,
                        top_n: int = 15) -> pd.DataFrame:
    
    output_dir = get_model_xai_dir(xai_name)

    importance = extract_feature_importance(model, X_reference, feature_names)

    structure = extract_tree_structure(model, X_reference, feature_names)

    save_dataframe(importance, output_dir / "impurity_importance.csv")

    save_dataframe(structure, output_dir / "tree_structure.csv")

    plot_feature_importance(importance, 
                            output_dir / "impurity_importance.png",
                            top_n=top_n)

    plot_tree_structure(model,
                        X_reference,
                        output_dir / "tree.png",
                        feature_names)

    logger.info("Zakończono globalne XAI drzewa: %s", xai_name)

    return importance



def run_local_tree_xai(model: TreeModel,
                       X_row: pd.DataFrame,
                       event_row: pd.Series,
                       xai_name: str,
                       feature_names: list[str] | None = None) -> pd.DataFrame:
    decision_path, summary = calculate_decision_path(model, X_row, feature_names)

    accession = str(event_row["Accession"])
    output_dir = get_local_xai_dir(xai_name, accession)

    save_dataframe(decision_path, output_dir / "decision_path.csv")

    event = build_event_metadata(event_row)

    if "XAI_Example_Type" in event_row.index:
        event["XAI_Example_Type"] = str(event_row["XAI_Example_Type"])

    save_json({"event": event, "prediction": summary},
              output_dir / "summary.json")

    logger.info("Zakończono local XAI drzewa | %s", accession)

    return decision_path