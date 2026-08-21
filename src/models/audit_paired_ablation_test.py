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

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "oos_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paired_ablation_test_logistic.csv"
)

N_PERMUTATIONS = 5000
RANDOM_STATE = 42


COMPARISONS = [
    (
        "SEC contribution: A -> B",
        "MODEL A - MARKET",
        "MODEL B - MARKET + SEC",
    ),
    (
        "FinBERT contribution: B -> C",
        "MODEL B - MARKET + SEC",
        "MODEL C - MARKET + SEC + FINBERT",
    ),
    (
    "Sentiment context contribution: C -> C2",
    "MODEL C - MARKET + SEC + FINBERT",
    "MODEL C2 - MARKET + SEC + FINBERT CONTEXT",
),
]


KEY_COLUMNS = [
    "Ticker",
    "Event_Session",
    "Accession",
    "Test_Year",
]


# ============================================================
# METRYKI
# ============================================================

def calculate_mean_yearly_metrics(
    df: pd.DataFrame,
    y_true_col: str,
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
            year_df[y_true_col]
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
        float(np.mean(balanced_scores)),
        float(np.mean(auc_scores)),
    )


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
# ŁĄCZENIE DWÓCH MODELI
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
            f"Brak modelu:\n{model_a}"
        )

    if right.empty:
        raise ValueError(
            f"Brak modelu:\n{model_b}"
        )

    # --------------------------------------------------------
    # Każdy model powinien mieć jeden rekord
    # na event OOS.
    # --------------------------------------------------------

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
            "Modele nie mają identycznego "
            "zbioru eventów OOS."
        )

    if len(merged) != len(right):
        raise ValueError(
            "Modele nie mają identycznego "
            "zbioru eventów OOS."
        )

    # --------------------------------------------------------
    # y_true MUSI być identyczne.
    # --------------------------------------------------------

    if not np.array_equal(
        merged["y_true_A"].to_numpy(),
        merged["y_true_B"].to_numpy(),
    ):
        raise ValueError(
            "Target różni się między modelami."
        )

    merged["y_true"] = (
        merged["y_true_A"]
        .astype(int)
    )

    return merged


# ============================================================
# TEST PAIRED PERMUTATION
# ============================================================

def run_paired_test(
    pair_df: pd.DataFrame,
    rng: np.random.Generator,
) -> dict:

    # --------------------------------------------------------
    # OBSERWOWANE METRYKI
    # --------------------------------------------------------

    metrics_a = (
        calculate_mean_yearly_metrics(
            pair_df,
            y_true_col="y_true",
            y_pred_col="y_pred_A",
            y_prob_col="y_prob_A",
        )
    )

    metrics_b = (
        calculate_mean_yearly_metrics(
            pair_df,
            y_true_col="y_true",
            y_pred_col="y_pred_B",
            y_prob_col="y_prob_B",
        )
    )

    balanced_a, auc_a = metrics_a
    balanced_b, auc_b = metrics_b

    observed_delta_balanced = (
        balanced_b
        - balanced_a
    )

    observed_delta_auc = (
        auc_b
        - auc_a
    )

    # --------------------------------------------------------
    # PERMUTACJE
    #
    # Dla każdego eventu losujemy, czy zamienić
    # predykcję modelu A z predykcją modelu B.
    #
    # Pod H0 modele są wymienne.
    # --------------------------------------------------------

    permutation_delta_balanced = np.empty(
        N_PERMUTATIONS,
        dtype=float,
    )

    permutation_delta_auc = np.empty(
        N_PERMUTATIONS,
        dtype=float,
    )

    years = sorted(
        pair_df["Test_Year"]
        .unique()
    )

    yearly_data = []

    for year in years:

        year_df = (
            pair_df[
                pair_df["Test_Year"]
                == year
            ]
            .copy()
        )

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

    for permutation_id in range(
        N_PERMUTATIONS
    ):

        balanced_a_years = []
        balanced_b_years = []

        auc_a_years = []
        auc_b_years = []

        for data in yearly_data:

            y_true = data["y_true"]

            pred_a = data["pred_a"]
            pred_b = data["pred_b"]

            prob_a = data["prob_a"]
            prob_b = data["prob_b"]

            # --------------------------------------------
            # Jeden niezależny swap na każdy event.
            # Ten sam swap stosujemy do klasy
            # i prawdopodobieństwa.
            # --------------------------------------------

            swap_mask = (
                rng.random(
                    len(y_true)
                )
                < 0.5
            )

            perm_pred_a = np.where(
                swap_mask,
                pred_b,
                pred_a,
            )

            perm_pred_b = np.where(
                swap_mask,
                pred_a,
                pred_b,
            )

            perm_prob_a = np.where(
                swap_mask,
                prob_b,
                prob_a,
            )

            perm_prob_b = np.where(
                swap_mask,
                prob_a,
                prob_b,
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

            auc_a_years.append(
                roc_auc_score(
                    y_true,
                    perm_prob_a,
                )
            )

            auc_b_years.append(
                roc_auc_score(
                    y_true,
                    perm_prob_b,
                )
            )

        perm_balanced_a = float(
            np.mean(
                balanced_a_years
            )
        )

        perm_balanced_b = float(
            np.mean(
                balanced_b_years
            )
        )

        perm_auc_a = float(
            np.mean(
                auc_a_years
            )
        )

        perm_auc_b = float(
            np.mean(
                auc_b_years
            )
        )

        permutation_delta_balanced[
            permutation_id
        ] = (
            perm_balanced_b
            - perm_balanced_a
        )

        permutation_delta_auc[
            permutation_id
        ] = (
            perm_auc_b
            - perm_auc_a
        )

    # --------------------------------------------------------
    # JEDNOSTRONNE P-VALUE
    #
    # H1: model B jest LEPSZY od modelu A.
    # --------------------------------------------------------

    p_balanced_one_sided = (
        1
        + np.sum(
            permutation_delta_balanced
            >= observed_delta_balanced
        )
    ) / (
        N_PERMUTATIONS + 1
    )

    p_auc_one_sided = (
        1
        + np.sum(
            permutation_delta_auc
            >= observed_delta_auc
        )
    ) / (
        N_PERMUTATIONS + 1
    )

    # --------------------------------------------------------
    # DWUSTRONNE P-VALUE
    #
    # Czy modele po prostu różnią się,
    # niezależnie od kierunku.
    # --------------------------------------------------------

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

    p_auc_two_sided = (
        1
        + np.sum(
            np.abs(
                permutation_delta_auc
            )
            >= abs(
                observed_delta_auc
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

        "Null_Delta_Balanced_Mean":
            float(
                np.mean(
                    permutation_delta_balanced
                )
            ),

        "Null_Delta_Balanced_Std":
            float(
                np.std(
                    permutation_delta_balanced,
                    ddof=1,
                )
            ),

        "P_Balanced_One_Sided":
            float(
                p_balanced_one_sided
            ),

        "P_Balanced_Two_Sided":
            float(
                p_balanced_two_sided
            ),

        "ROC_A":
            auc_a,

        "ROC_B":
            auc_b,

        "Delta_ROC":
            observed_delta_auc,

        "Null_Delta_ROC_Mean":
            float(
                np.mean(
                    permutation_delta_auc
                )
            ),

        "Null_Delta_ROC_Std":
            float(
                np.std(
                    permutation_delta_auc,
                    ddof=1,
                )
            ),

        "P_ROC_One_Sided":
            float(
                p_auc_one_sided
            ),

        "P_ROC_Two_Sided":
            float(
                p_auc_two_sided
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
        "PAIRED ABLATION TEST - LOGISTIC REGRESSION"
    )

    print(
        "=" * 80
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Nie znaleziono:\n"
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        "\nPlik:"
    )
    print(
        INPUT_FILE
    )

    print(
        "\nLiczba wierszy:"
    )
    print(
        len(df)
    )

    print(
        "\nPredykcje per model:"
    )
    print(
        df["Model"]
        .value_counts()
        .sort_index()
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    results = []

    for (
        comparison_name,
        model_a,
        model_b,
    ) in COMPARISONS:

        print(
            "\n"
            + "-" * 80
        )

        print(
            comparison_name
        )

        print(
            "\nModel bazowy:"
        )
        print(
            model_a
        )

        print(
            "\nModel rozszerzony:"
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
            rng=rng,
        )

        result["Comparison"] = (
            comparison_name
        )

        result["Model_A"] = model_a
        result["Model_B"] = model_b

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
            "p two-sided:",
            round(
                result[
                    "P_Balanced_Two_Sided"
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

        print(
            "p two-sided:",
            round(
                result[
                    "P_ROC_Two_Sided"
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

    # Korekta Holma:
    # 2 porównania dla Balanced Accuracy.
    results_df[
        "P_Balanced_One_Sided_Holm"
    ] = holm_adjust(
        results_df[
            "P_Balanced_One_Sided"
        ].tolist()
    )

    # 2 porównania dla ROC-AUC.
    results_df[
        "P_ROC_One_Sided_Holm"
    ] = holm_adjust(
        results_df[
            "P_ROC_One_Sided"
        ].tolist()
    )

    columns = [
        "Comparison",
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

    results_df = results_df[
        columns
    ]

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