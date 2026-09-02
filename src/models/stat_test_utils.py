import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


KEY_COLUMNS = ["Ticker", "Event_Session", "Accession", "Test_Year"]
PAIR_KEY_COLUMNS = ["Ticker", "Event_Session", "Test_Year"]


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running_max = 0.0

    for rank, index in enumerate(order):
        running_max = max(running_max, (len(p) - rank) * p[index])
        adjusted[index] = min(running_max, 1.0)

    return adjusted.tolist()


def classification_metrics(df: pd.DataFrame, pred_col: str, prob_col: str) -> tuple[float, float]:
    y_true = df["y_true"].astype(int).to_numpy()
    if np.unique(y_true).size != 2:
        raise ValueError("ROC-AUC wymaga obu klas w całym zbiorze OOS.")

    balanced = balanced_accuracy_score(y_true, df[pred_col].astype(int).to_numpy())
    auc = roc_auc_score(y_true, df[prob_col].astype(float).to_numpy())
    return float(balanced), float(auc)


def prepare_prediction_pair(df: pd.DataFrame, model_a: str, model_b: str) -> pd.DataFrame:
    required = {"Model", *KEY_COLUMNS, "y_true", "y_pred", "y_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Brak wymaganych kolumn: {sorted(missing)}")

    def aggregate_model(model_name: str) -> pd.DataFrame:
        model_df = df.loc[df["Model"].eq(model_name)].copy()
        consistency = model_df.groupby(PAIR_KEY_COLUMNS)["y_true"].nunique()
        if consistency.gt(1).any():
            raise ValueError(f"{model_name}: ten sam ticker i sesja mają różny target.")

        events = model_df.groupby(PAIR_KEY_COLUMNS, as_index=False).agg(
            Accession=("Accession", lambda values: " | ".join(sorted(values.astype(str)))),
            y_true=("y_true", "first"), y_prob=("y_prob", "mean"))
        events["y_pred"] = events["y_prob"].ge(0.5).astype(int)
        return events

    values = ["Accession", "y_true", "y_pred", "y_prob"]
    left = aggregate_model(model_a).set_index(PAIR_KEY_COLUMNS)[values].add_suffix("_A")
    right = aggregate_model(model_b).set_index(PAIR_KEY_COLUMNS)[values].add_suffix("_B")

    if left.empty:
        raise ValueError(f"Nie znaleziono modelu A: {model_a}")
    if right.empty:
        raise ValueError(f"Nie znaleziono modelu B: {model_b}")
    if not left.index.is_unique or not right.index.is_unique:
        raise ValueError("Znaleziono duplikaty eventów w predykcjach.")

    pair_df = left.join(right, how="inner", validate="one_to_one")
    if len(pair_df) != len(left) or len(pair_df) != len(right):
        raise ValueError("Porównywane modele nie mają identycznego zbioru OOS.")
    if not np.array_equal(pair_df["y_true_A"].to_numpy(), pair_df["y_true_B"].to_numpy()):
        raise ValueError("Target różni się między porównywanymi modelami.")
    if not pair_df["Accession_A"].equals(pair_df["Accession_B"]):
        raise ValueError("Lista filingów różni się między porównywanymi modelami.")

    pair_df["y_true"] = pair_df["y_true_A"].astype(int)
    pair_df["Accession"] = pair_df["Accession_A"]
    return pair_df.reset_index().sort_values(KEY_COLUMNS).reset_index(drop=True)


def run_paired_permutation_test(pair_df: pd.DataFrame, n_permutations: int, random_state: int) -> dict:
    """Losuje zamianę modeli całymi sesjami, zachowując zależność eventów z jednego dnia."""
    rng = np.random.default_rng(random_state)
    balanced_a, auc_a = classification_metrics(pair_df, "y_pred_A", "y_prob_A")
    balanced_b, auc_b = classification_metrics(pair_df, "y_pred_B", "y_prob_B")
    delta_balanced, delta_auc = balanced_b - balanced_a, auc_b - auc_a

    cluster_codes, clusters = pd.factorize(pair_df["Event_Session"], sort=True)
    y_true = pair_df["y_true"].astype(int).to_numpy()
    pred_a, pred_b = pair_df["y_pred_A"].astype(int).to_numpy(), pair_df["y_pred_B"].astype(int).to_numpy()
    prob_a, prob_b = pair_df["y_prob_A"].astype(float).to_numpy(), pair_df["y_prob_B"].astype(float).to_numpy()
    permutation_delta_balanced = np.empty(n_permutations)
    permutation_delta_auc = np.empty(n_permutations)

    for permutation_id in range(n_permutations):
        cluster_swap = rng.random(len(clusters)) < 0.5
        swap = cluster_swap[cluster_codes]
        perm_pred_a, perm_pred_b = np.where(swap, pred_b, pred_a), np.where(swap, pred_a, pred_b)
        perm_prob_a, perm_prob_b = np.where(swap, prob_b, prob_a), np.where(swap, prob_a, prob_b)
        perm_balanced_a = balanced_accuracy_score(y_true, perm_pred_a)
        perm_balanced_b = balanced_accuracy_score(y_true, perm_pred_b)
        perm_auc_a, perm_auc_b = roc_auc_score(y_true, perm_prob_a), roc_auc_score(y_true, perm_prob_b)
        permutation_delta_balanced[permutation_id] = perm_balanced_b - perm_balanced_a
        permutation_delta_auc[permutation_id] = perm_auc_b - perm_auc_a

    p_balanced = (1 + np.sum(permutation_delta_balanced >= delta_balanced)) / (n_permutations + 1)
    p_auc = (1 + np.sum(permutation_delta_auc >= delta_auc)) / (n_permutations + 1)

    return {"N_OOS": len(pair_df), "N_Sessions": len(clusters),
        "N_Pred_Diff": int(pair_df["y_pred_A"].ne(pair_df["y_pred_B"]).sum()),
        "Pooled_BA_A": balanced_a, "Pooled_BA_B": balanced_b, "Delta_BA": delta_balanced,
        "Null_Delta_BA_Mean": float(permutation_delta_balanced.mean()), "P_BA": float(p_balanced),
        "Pooled_AUC_A": auc_a, "Pooled_AUC_B": auc_b, "Delta_AUC": delta_auc,
        "Null_Delta_AUC_Mean": float(permutation_delta_auc.mean()), "P_AUC": float(p_auc)}
