from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
)


# ============================================================
# USTAWIENIA
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

N_PERMUTATIONS = 5000
RANDOM_STATE = 42

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sec_robustness_test.csv"
)

KEY_COLUMNS = [
    "Ticker",
    "Event_Session",
    "Accession",
    "Test_Year",
]


EXPERIMENTS = [
    {
        "Algorithm": "Logistic Regression",
        "File": (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "oos_predictions.csv"
        ),
        "Model_A": "MODEL A - MARKET",
        "Model_B": "MODEL B - MARKET + SEC",
    },
    {
        "Algorithm": "XGBoost",
        "File": (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "xgboost_oos_predictions.csv"
        ),
        "Model_A": "XGB A - COMPACT MARKET",
        "Model_B": "XGB B - COMPACT + SEC",
    },
    {
        "Algorithm": "CatBoost",
        "File": (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "catboost_oos_predictions.csv"
        ),
        "Model_A": "CAT A - COMPACT MARKET",
        "Model_B": "CAT B - COMPACT + SEC",
    },
]


# ============================================================
# HOLM
# ============================================================

def holm_adjust(
    p_values: list[float],
) -> list[float]:

    p = np.asarray(
        p_values,
        dtype=float,
    )

    n = len(p)

    order = np.argsort(p)

    adjusted = np.empty(
        n,
        dtype=float,
    )

    running_max = 0.0

    for rank, index in enumerate(order):

        value = (
            (n - rank)
            * p[index]
        )

        running_max = max(
            running_max,
            value,
        )

        adjusted[index] = min(
            running_max,
            1.0,
        )

    return adjusted.tolist()


# ============================================================
# ŚREDNIE METRYKI PO ROCZNYCH FOLDACH
# ============================================================

def calculate_mean_yearly_metrics(
    df: pd.DataFrame,
    y_pred_col: str,
    y_prob_col: str,
) -> tuple[float, float]:

    balanced_scores = []
    auc_scores = []

    for _, year_df in df.groupby(
        "Test_Year",
        sort=True,
    ):

        y_true = (
            year_df["y_true"]
            .astype(int)
            .to_numpy()
        )

        y_pred = (
            year_df[y_pred_col]
            .astype(int)
            .to_numpy()
        )

        y_prob = (
            year_df[y_prob_col]
            .astype(float)
            .to_numpy()
        )

        balanced_scores.append(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        )

        if np.unique(y_true).size == 2:
            auc_scores.append(
                roc_auc_score(
                    y_true,
                    y_prob,
                )
            )

    return (
        float(
            np.mean(
                balanced_scores
            )
        ),
        float(
            np.mean(
                auc_scores
            )
        ),
    )


# ============================================================
# PRZYGOTOWANIE PARY A / B
# ============================================================

def prepare_pair(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
) -> pd.DataFrame:

    left = (
        df[
            df["Model"] == model_a
        ]
        .copy()
    )

    right = (
        df[
            df["Model"] == model_b
        ]
        .copy()
    )

    if left.empty:
        raise ValueError(
            f"Nie znaleziono modelu A:\n{model_a}"
        )

    if right.empty:
        raise ValueError(
            f"Nie znaleziono modelu B:\n{model_b}"
        )

    if left.duplicated(
        subset=KEY_COLUMNS
    ).any():
        raise ValueError(
            f"Duplikaty w modelu:\n{model_a}"
        )

    if right.duplicated(
        subset=KEY_COLUMNS
    ).any():
        raise ValueError(
            f"Duplikaty w modelu:\n{model_b}"
        )

    left = left[
        KEY_COLUMNS
        + [
            "y_true",
            "y_pred",
            "y_prob",
        ]
    ].rename(
        columns={
            "y_true": "y_true_A",
            "y_pred": "y_pred_A",
            "y_prob": "y_prob_A",
        }
    )

    right = right[
        KEY_COLUMNS
        + [
            "y_true",
            "y_pred",
            "y_prob",
        ]
    ].rename(
        columns={
            "y_true": "y_true_B",
            "y_pred": "y_pred_B",
            "y_prob": "y_prob_B",
        }
    )

    merged = left.merge(
        right,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(left):
        raise ValueError(
            "Modele A i B nie mają "
            "identycznych eventów."
        )

    if len(merged) != len(right):
        raise ValueError(
            "Modele A i B nie mają "
            "identycznych eventów."
        )

    if not np.array_equal(
        merged["y_true_A"].to_numpy(),
        merged["y_true_B"].to_numpy(),
    ):
        raise ValueError(
            "Target różni się między A i B."
        )

    merged["y_true"] = (
        merged["y_true_A"]
        .astype(int)
    )

    return merged


# ============================================================
# PAIRED PERMUTATION TEST
# ============================================================

def run_paired_test(
    pair_df: pd.DataFrame,
    random_state: int,
) -> dict:

    rng = np.random.default_rng(
        random_state
    )

    balanced_a, roc_a = (
        calculate_mean_yearly_metrics(
            pair_df,
            y_pred_col="y_pred_A",
            y_prob_col="y_prob_A",
        )
    )

    balanced_b, roc_b = (
        calculate_mean_yearly_metrics(
            pair_df,
            y_pred_col="y_pred_B",
            y_prob_col="y_prob_B",
        )
    )

    observed_delta_balanced = (
        balanced_b
        - balanced_a
    )

    observed_delta_roc = (
        roc_b
        - roc_a
    )

    permutation_delta_balanced = np.empty(
        N_PERMUTATIONS,
        dtype=float,
    )

    permutation_delta_roc = np.empty(
        N_PERMUTATIONS,
        dtype=float,
    )

    yearly_data = []

    for _, year_df in pair_df.groupby(
        "Test_Year",
        sort=True,
    ):

        yearly_data.append(
            {
                "y_true":
                    year_df["y_true"]
                    .astype(int)
                    .to_numpy(),

                "pred_a":
                    year_df["y_pred_A"]
                    .astype(int)
                    .to_numpy(),

                "pred_b":
                    year_df["y_pred_B"]
                    .astype(int)
                    .to_numpy(),

                "prob_a":
                    year_df["y_prob_A"]
                    .astype(float)
                    .to_numpy(),

                "prob_b":
                    year_df["y_prob_B"]
                    .astype(float)
                    .to_numpy(),
            }
        )

    # ========================================================
    # PERMUTACJE
    # ========================================================

    for permutation_id in range(
        N_PERMUTATIONS
    ):

        balanced_a_years = []
        balanced_b_years = []

        roc_a_years = []
        roc_b_years = []

        for data in yearly_data:

            y_true = data["y_true"]

            swap_mask = (
                rng.random(
                    len(y_true)
                )
                < 0.5
            )

            perm_pred_a = np.where(
                swap_mask,
                data["pred_b"],
                data["pred_a"],
            )

            perm_pred_b = np.where(
                swap_mask,
                data["pred_a"],
                data["pred_b"],
            )

            perm_prob_a = np.where(
                swap_mask,
                data["prob_b"],
                data["prob_a"],
            )

            perm_prob_b = np.where(
                swap_mask,
                data["prob_a"],
                data["prob_b"],
            )

            balanced_a_years.append(
                balanced_accuracy_score(
                    y_true,
                    perm_pred_a,
                )
            )

            balanced_b_years.append(
                balanced_accuracy_score(
                    y_true,
                    perm_pred_b,
                )
            )

            roc_a_years.append(
                roc_auc_score(
                    y_true,
                    perm_prob_a,
                )
            )

            roc_b_years.append(
                roc_auc_score(
                    y_true,
                    perm_prob_b,
                )
            )

        permutation_delta_balanced[
            permutation_id
        ] = (
            np.mean(
                balanced_b_years
            )
            -
            np.mean(
                balanced_a_years
            )
        )

        permutation_delta_roc[
            permutation_id
        ] = (
            np.mean(
                roc_b_years
            )
            -
            np.mean(
                roc_a_years
            )
        )

    # ========================================================
    # JEDNOSTRONNE:
    # H1 = dodanie SEC poprawia wynik
    # ========================================================

    p_balanced_one_sided = (
        1
        + np.sum(
            permutation_delta_balanced
            >= observed_delta_balanced
        )
    ) / (
        N_PERMUTATIONS + 1
    )

    p_roc_one_sided = (
        1
        + np.sum(
            permutation_delta_roc
            >= observed_delta_roc
        )
    ) / (
        N_PERMUTATIONS + 1
    )

    # ========================================================
    # DWUSTRONNE
    # ========================================================

    p_balanced_two_sided = (
        1
        + np.sum(
            np.abs(
                permutation_delta_balanced
            )
            >= abs(
                observed_delta_balanced
            )
        )
    ) / (
        N_PERMUTATIONS + 1
    )

    p_roc_two_sided = (
        1
        + np.sum(
            np.abs(
                permutation_delta_roc
            )
            >= abs(
                observed_delta_roc
            )
        )
    ) / (
        N_PERMUTATIONS + 1
    )

    return {
        "N_OOS":
            len(pair_df),

        "Balanced_A":
            balanced_a,

        "Balanced_B":
            balanced_b,

        "Delta_Balanced":
            observed_delta_balanced,

        "P_Balanced_One_Sided":
            float(
                p_balanced_one_sided
            ),

        "P_Balanced_Two_Sided":
            float(
                p_balanced_two_sided
            ),

        "ROC_A":
            roc_a,

        "ROC_B":
            roc_b,

        "Delta_ROC":
            observed_delta_roc,

        "P_ROC_One_Sided":
            float(
                p_roc_one_sided
            ),

        "P_ROC_Two_Sided":
            float(
                p_roc_two_sided
            ),
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 80
    )

    print(
        "SEC ROBUSTNESS TEST"
    )

    print(
        "=" * 80
    )

    print(
        "\nLiczba permutacji:",
        N_PERMUTATIONS,
    )

    results = []

    for experiment_id, experiment in enumerate(
        EXPERIMENTS
    ):

        algorithm = experiment[
            "Algorithm"
        ]

        input_file = experiment[
            "File"
        ]

        model_a = experiment[
            "Model_A"
        ]

        model_b = experiment[
            "Model_B"
        ]

        print(
            "\n"
            + "-" * 80
        )

        print(
            algorithm
        )

        print(
            "-" * 80
        )

        if not input_file.exists():
            raise FileNotFoundError(
                f"Nie znaleziono:\n"
                f"{input_file}"
            )

        df = pd.read_csv(
            input_file
        )

        print(
            "\nPlik:"
        )

        print(
            input_file
        )

        print(
            "\nModel A:"
        )

        print(
            model_a
        )

        print(
            "\nModel B:"
        )

        print(
            model_b
        )

        pair_df = prepare_pair(
            df=df,
            model_a=model_a,
            model_b=model_b,
        )

        print(
            "\nLiczba sparowanych eventów:"
        )

        print(
            len(pair_df)
        )

        result = run_paired_test(
            pair_df=pair_df,
            random_state=(
                RANDOM_STATE
                + experiment_id
            ),
        )

        result[
            "Algorithm"
        ] = algorithm

        result[
            "Model_A"
        ] = model_a

        result[
            "Model_B"
        ] = model_b

        results.append(
            result
        )

        print(
            "\nBalanced Accuracy:"
        )

        print(
            "A:",
            round(
                result["Balanced_A"],
                6,
            ),
        )

        print(
            "B:",
            round(
                result["Balanced_B"],
                6,
            ),
        )

        print(
            "Delta:",
            round(
                result["Delta_Balanced"],
                6,
            ),
        )

        print(
            "p one-sided:",
            round(
                result[
                    "P_Balanced_One_Sided"
                ],
                6,
            ),
        )

        print(
            "\nROC-AUC:"
        )

        print(
            "A:",
            round(
                result["ROC_A"],
                6,
            ),
        )

        print(
            "B:",
            round(
                result["ROC_B"],
                6,
            ),
        )

        print(
            "Delta:",
            round(
                result["Delta_ROC"],
                6,
            ),
        )

        print(
            "p one-sided:",
            round(
                result[
                    "P_ROC_One_Sided"
                ],
                6,
            ),
        )

    # ========================================================
    # PODSUMOWANIE
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    # Korekta Holma osobno:
    # 3 algorytmy dla Balanced Accuracy
    # i 3 algorytmy dla ROC-AUC.
    results_df[
        "P_Balanced_One_Sided_Holm"
    ] = holm_adjust(
        results_df[
            "P_Balanced_One_Sided"
        ].tolist()
    )

    results_df[
        "P_ROC_One_Sided_Holm"
    ] = holm_adjust(
        results_df[
            "P_ROC_One_Sided"
        ].tolist()
    )

    columns = [
        "Algorithm",
        "N_OOS",

        "Balanced_A",
        "Balanced_B",
        "Delta_Balanced",
        "P_Balanced_One_Sided",
        "P_Balanced_One_Sided_Holm",
        "P_Balanced_Two_Sided",

        "ROC_A",
        "ROC_B",
        "Delta_ROC",
        "P_ROC_One_Sided",
        "P_ROC_One_Sided_Holm",
        "P_ROC_Two_Sided",
    ]

    results_df = (
        results_df[
            columns
        ]
        .copy()
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "PODSUMOWANIE"
    )

    print(
        "=" * 80
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        "\nWyniki zapisano do:"
    )

    print(
        OUTPUT_FILE
    )