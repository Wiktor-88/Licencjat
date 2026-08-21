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
    / "permutation_test_logistic.csv"
)

N_PERMUTATIONS = 5000
RANDOM_STATE = 42


MODELS = [
    "MODEL A - MARKET",
    "MODEL B - MARKET + SEC",
    "MODEL C - MARKET + SEC + FINBERT",
]


# ============================================================
# POMOCNICZE
# ============================================================

def find_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str:
    """
    Znajduje pierwszą istniejącą kolumnę
    spośród podanych kandydatów.
    """

    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        "Nie znaleziono żadnej z kolumn:\n"
        + "\n".join(candidates)
        + "\n\nDostępne kolumny:\n"
        + "\n".join(df.columns)
    )


def calculate_mean_yearly_metrics(
    df: pd.DataFrame,
    year_col: str,
    y_true_col: str,
    y_pred_col: str,
    y_prob_col: str,
) -> tuple[float, float]:
    """
    Liczy średnią metrykę po rocznych
    foldach walk-forward.

    Dzięki temu statystyka jest zgodna
    z dotychczasowym raportowaniem modeli.
    """

    balanced_scores = []
    auc_scores = []

    for _, year_df in df.groupby(
        year_col,
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

    mean_balanced = float(
        np.mean(balanced_scores)
    )

    mean_auc = float(
        np.mean(auc_scores)
    )

    return (
        mean_balanced,
        mean_auc,
    )


def holm_adjust(
    p_values: list[float],
) -> list[float]:
    """
    Korekta Holma dla wielokrotnych testów.
    """

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
# TEST PERMUTACYJNY
# ============================================================

def run_permutation_test(
    model_df: pd.DataFrame,
    year_col: str,
    y_true_col: str,
    y_pred_col: str,
    y_prob_col: str,
    rng: np.random.Generator,
) -> dict:

    observed_balanced, observed_auc = (
        calculate_mean_yearly_metrics(
            model_df,
            year_col,
            y_true_col,
            y_pred_col,
            y_prob_col,
        )
    )

    permutation_balanced = np.empty(
        N_PERMUTATIONS,
        dtype=float,
    )

    permutation_auc = np.empty(
        N_PERMUTATIONS,
        dtype=float,
    )

    # --------------------------------------------------------
    # Przygotowujemy dane każdego roku osobno.
    # --------------------------------------------------------

    yearly_data = []

    for year, year_df in model_df.groupby(
        year_col,
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

        yearly_data.append(
            (
                year,
                y_true,
                y_pred,
                y_prob,
            )
        )

    # --------------------------------------------------------
    # Permutacje
    # --------------------------------------------------------

    for permutation_id in range(
        N_PERMUTATIONS
    ):

        balanced_scores = []
        auc_scores = []

        for (
            _,
            y_true,
            y_pred,
            y_prob,
        ) in yearly_data:

            # Kluczowy element:
            # mieszamy target tylko wewnątrz roku.
            y_permuted = rng.permutation(
                y_true
            )

            balanced_scores.append(
                balanced_accuracy_score(
                    y_permuted,
                    y_pred,
                )
            )

            if np.unique(y_permuted).size == 2:
                auc_scores.append(
                    roc_auc_score(
                        y_permuted,
                        y_prob,
                    )
                )

        permutation_balanced[
            permutation_id
        ] = np.mean(
            balanced_scores
        )

        permutation_auc[
            permutation_id
        ] = np.mean(
            auc_scores
        )

    # --------------------------------------------------------
    # Jednostronne empirical p-value:
    #
    # H0: model nie ma przewagi nad przypadkiem
    # H1: metryka modelu jest większa
    # --------------------------------------------------------

    p_balanced = (
        1
        + np.sum(
            permutation_balanced
            >= observed_balanced
        )
    ) / (
        N_PERMUTATIONS + 1
    )

    p_auc = (
        1
        + np.sum(
            permutation_auc
            >= observed_auc
        )
    ) / (
        N_PERMUTATIONS + 1
    )

    result = {
        "Observed_Balanced_Accuracy":
            observed_balanced,

        "Null_Balanced_Mean":
            float(
                np.mean(
                    permutation_balanced
                )
            ),

        "Null_Balanced_Std":
            float(
                np.std(
                    permutation_balanced,
                    ddof=1,
                )
            ),

        "Null_Balanced_95":
            float(
                np.quantile(
                    permutation_balanced,
                    0.95,
                )
            ),

        "P_Value_Balanced":
            float(p_balanced),

        "Observed_ROC_AUC":
            observed_auc,

        "Null_ROC_Mean":
            float(
                np.mean(
                    permutation_auc
                )
            ),

        "Null_ROC_Std":
            float(
                np.std(
                    permutation_auc,
                    ddof=1,
                )
            ),

        "Null_ROC_95":
            float(
                np.quantile(
                    permutation_auc,
                    0.95,
                )
            ),

        "P_Value_ROC":
            float(p_auc),
    }

    return result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 80
    )

    print(
        "PERMUTATION TEST - LOGISTIC REGRESSION OOS"
    )

    print(
        "=" * 80
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku:\n"
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
        "\nKolumny:"
    )
    print(
        df.columns.tolist()
    )

    # --------------------------------------------------------
    # Automatyczne znalezienie nazw kolumn
    # --------------------------------------------------------

    model_col = find_column(
        df,
        [
            "Model",
            "model",
        ],
    )

    year_col = find_column(
        df,
        [
            "Test_Year",
            "Year",
            "test_year",
        ],
    )

    y_true_col = find_column(
        df,
        [
            "y_true",
            "Y_True",
            "Target",
            "Target_Abnormal_1D",
        ],
    )

    y_pred_col = find_column(
        df,
        [
            "y_pred",
            "Y_Pred",
            "Prediction",
        ],
    )

    y_prob_col = find_column(
        df,
        [
            "y_prob",
            "Y_Prob",
            "Probability",
            "Prob_1",
        ],
    )

    print(
        "\nUżyte kolumny:"
    )

    print(
        f"Model:  {model_col}"
    )
    print(
        f"Year:   {year_col}"
    )
    print(
        f"y_true: {y_true_col}"
    )
    print(
        f"y_pred: {y_pred_col}"
    )
    print(
        f"y_prob: {y_prob_col}"
    )

    print(
        f"\nLiczba permutacji: "
        f"{N_PERMUTATIONS}"
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    results = []

    for model_name in MODELS:

        print(
            "\n"
            + "-" * 80
        )

        print(
            model_name
        )

        model_df = df[
            df[model_col]
            == model_name
        ].copy()

        if model_df.empty:
            raise ValueError(
                "Nie znaleziono modelu:\n"
                f"{model_name}\n\n"
                "Dostępne modele:\n"
                + "\n".join(
                    sorted(
                        df[model_col]
                        .dropna()
                        .unique()
                    )
                )
            )

        print(
            "Liczba predykcji OOS:",
            len(model_df),
        )

        result = run_permutation_test(
            model_df=model_df,
            year_col=year_col,
            y_true_col=y_true_col,
            y_pred_col=y_pred_col,
            y_prob_col=y_prob_col,
            rng=rng,
        )

        result["Model"] = model_name

        results.append(
            result
        )

        print(
            "\nBalanced Accuracy:"
        )

        print(
            "Observed:",
            round(
                result[
                    "Observed_Balanced_Accuracy"
                ],
                6,
            ),
        )

        print(
            "Null mean:",
            round(
                result[
                    "Null_Balanced_Mean"
                ],
                6,
            ),
        )

        print(
            "Null 95%:",
            round(
                result[
                    "Null_Balanced_95"
                ],
                6,
            ),
        )

        print(
            "p-value:",
            round(
                result[
                    "P_Value_Balanced"
                ],
                6,
            ),
        )

        print(
            "\nROC-AUC:"
        )

        print(
            "Observed:",
            round(
                result[
                    "Observed_ROC_AUC"
                ],
                6,
            ),
        )

        print(
            "Null mean:",
            round(
                result[
                    "Null_ROC_Mean"
                ],
                6,
            ),
        )

        print(
            "Null 95%:",
            round(
                result[
                    "Null_ROC_95"
                ],
                6,
            ),
        )

        print(
            "p-value:",
            round(
                result[
                    "P_Value_ROC"
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

    # Korekta Holma osobno dla dwóch rodzin testów.
    results_df[
        "P_Value_Balanced_Holm"
    ] = holm_adjust(
        results_df[
            "P_Value_Balanced"
        ].tolist()
    )

    results_df[
        "P_Value_ROC_Holm"
    ] = holm_adjust(
        results_df[
            "P_Value_ROC"
        ].tolist()
    )

    column_order = [
        "Model",

        "Observed_Balanced_Accuracy",
        "Null_Balanced_Mean",
        "Null_Balanced_Std",
        "Null_Balanced_95",
        "P_Value_Balanced",
        "P_Value_Balanced_Holm",

        "Observed_ROC_AUC",
        "Null_ROC_Mean",
        "Null_ROC_Std",
        "Null_ROC_95",
        "P_Value_ROC",
        "P_Value_ROC_Holm",
    ]

    results_df = results_df[
        column_order
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