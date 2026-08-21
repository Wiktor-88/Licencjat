from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# ŚCIEŻKI
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "oos_predictions.csv"
)

MARKET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "market_features.csv"
)

OUTPUT_OVERALL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "financial_metrics_overall.csv"
)

OUTPUT_YEARLY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "financial_metrics_by_year.csv"
)

OUTPUT_EVENTS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "financial_event_returns.csv"
)

OUTPUT_DAILY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "financial_daily_returns.csv"
)


# ============================================================
# PARAMETRY BACKTESTU
# ============================================================

PROBABILITY_THRESHOLD = 0.50

# Na pierwszym etapie ignorujemy koszty.
# Później zrobimy sensitivity analysis np. 5 / 10 bps.
TRANSACTION_COST_BPS = 0.0

TRADING_DAYS_PER_YEAR = 252


# ============================================================
# WALIDACJA
# ============================================================

def validate_predictions(
    df: pd.DataFrame,
) -> None:

    required_columns = [
        "Ticker",
        "Event_Session",
        "Accession",
        "Abnormal_Event_Return_1D",
        "Test_Year",
        "Model",
        "y_true",
        "y_pred",
        "y_prob",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Brakuje kolumn w oos_predictions.csv:\n"
            + "\n".join(missing)
        )


# ============================================================
# 1 TICKER + 1 EVENT SESSION = 1 TRANSAKCJA
# ============================================================

def aggregate_filing_predictions(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Jeżeli kilka filingów tego samego tickera prowadzi
    do tego samego Event_Session, tworzymy jedną decyzję.

    Prawdopodobieństwa są uśredniane.
    """

    # Najpierw sprawdzamy, czy abnormal return
    # jest identyczny wewnątrz tego samego okna eventowego.

    return_check = (
        df.groupby(
            [
                "Model",
                "Ticker",
                "Event_Session",
            ]
        )["Abnormal_Event_Return_1D"]
        .nunique()
    )

    invalid = return_check[
        return_check > 1
    ]

    if not invalid.empty:
        raise ValueError(
            "Ten sam Ticker + Event_Session ma "
            "różne abnormal returns:\n"
            f"{invalid}"
        )

    events = (
        df.groupby(
            [
                "Model",
                "Ticker",
                "Event_Session",
            ],
            as_index=False,
        )
        .agg(
            Test_Year=(
                "Test_Year",
                "first",
            ),
            Filing_Count=(
                "Accession",
                "size",
            ),
            y_prob=(
                "y_prob",
                "mean",
            ),
            Abnormal_Event_Return_1D=(
                "Abnormal_Event_Return_1D",
                "first",
            ),
        )
    )

    # ========================================================
    # SYGNAŁ
    # ========================================================

    events["Signal"] = np.where(
        events["y_prob"]
        >= PROBABILITY_THRESHOLD,
        1,
        -1,
    )

    # ========================================================
    # ZWROT STRATEGII
    # ========================================================

    events["Strategy_Return_Gross"] = (
        events["Signal"]
        * events["Abnormal_Event_Return_1D"]
    )

    transaction_cost = (
        TRANSACTION_COST_BPS
        / 10000.0
    )

    events["Strategy_Return"] = (
        events["Strategy_Return_Gross"]
        - transaction_cost
    )

    events["Win"] = (
        events["Strategy_Return"] > 0
    ).astype(int)

    return events


# ============================================================
# PORTFEL NA POZIOMIE SESJI
# ============================================================

def build_session_returns(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Jeżeli tego samego dnia kilka spółek daje sygnał,
    traktujemy je jako równoważony portfel.

    Dzięki temu nie kapitalizujemy np. trzech pełnych
    pozycji jedna po drugiej w tej samej sesji.
    """

    sessions = (
        events.groupby(
            [
                "Model",
                "Event_Session",
            ],
            as_index=False,
        )
        .agg(
            Strategy_Return=(
                "Strategy_Return",
                "mean",
            ),
            Num_Positions=(
                "Ticker",
                "nunique",
            ),
        )
    )

    sessions["Test_Year"] = (
        sessions["Event_Session"]
        .dt.year
    )

    return sessions


# ============================================================
# PEŁNY KALENDARZ SESJI
# ============================================================

def build_daily_returns(
    model_sessions: pd.DataFrame,
    trading_calendar: pd.DataFrame,
) -> pd.DataFrame:

    start_date = (
        model_sessions["Event_Session"]
        .min()
    )

    end_date = (
        model_sessions["Event_Session"]
        .max()
    )

    calendar = trading_calendar[
        (
            trading_calendar["Date"]
            >= start_date
        )
        & (
            trading_calendar["Date"]
            <= end_date
        )
    ].copy()

    calendar = calendar.rename(
        columns={
            "Date": "Event_Session",
        }
    )

    daily = calendar.merge(
        model_sessions[
            [
                "Event_Session",
                "Strategy_Return",
                "Num_Positions",
            ]
        ],
        on="Event_Session",
        how="left",
    )

    # W dni bez sygnału strategia pozostaje w gotówce.
    daily["Strategy_Return"] = (
        daily["Strategy_Return"]
        .fillna(0.0)
    )

    daily["Num_Positions"] = (
        daily["Num_Positions"]
        .fillna(0)
        .astype(int)
    )

    daily["Equity"] = (
        1.0
        + daily["Strategy_Return"]
    ).cumprod()

    daily["Running_Max"] = (
        daily["Equity"]
        .cummax()
    )

    daily["Drawdown"] = (
        daily["Equity"]
        / daily["Running_Max"]
        - 1.0
    )

    return daily


# ============================================================
# METRYKI FINANSOWE
# ============================================================

def calculate_financial_metrics(
    events: pd.DataFrame,
    daily: pd.DataFrame,
) -> dict:

    event_returns = (
        events["Strategy_Return"]
    )

    daily_returns = (
        daily["Strategy_Return"]
    )

    wins = event_returns[
        event_returns > 0
    ]

    losses = event_returns[
        event_returns < 0
    ]

    # ========================================================
    # EXPECTED VALUE
    # ========================================================

    expected_value = (
        event_returns.mean()
    )

    win_rate = (
        (event_returns > 0)
        .mean()
    )

    avg_win = (
        wins.mean()
        if len(wins) > 0
        else np.nan
    )

    avg_loss = (
        abs(losses.mean())
        if len(losses) > 0
        else np.nan
    )

    # ========================================================
    # PROFIT FACTOR
    # ========================================================

    gross_profit = (
        wins.sum()
    )

    gross_loss = (
        abs(losses.sum())
    )

    if gross_loss > 0:
        profit_factor = (
            gross_profit
            / gross_loss
        )
    else:
        profit_factor = np.inf

    # ========================================================
    # SHARPE
    # ========================================================

    daily_std = (
        daily_returns.std(
            ddof=1
        )
    )

    if (
        pd.notna(daily_std)
        and daily_std > 0
    ):
        sharpe = (
            daily_returns.mean()
            / daily_std
            * np.sqrt(
                TRADING_DAYS_PER_YEAR
            )
        )
    else:
        sharpe = np.nan

    # ========================================================
    # SORTINO
    # ========================================================

    downside = np.minimum(
        daily_returns.to_numpy(),
        0.0,
    )

    downside_deviation = np.sqrt(
        np.mean(
            downside ** 2
        )
    )

    if downside_deviation > 0:
        sortino = (
            daily_returns.mean()
            / downside_deviation
            * np.sqrt(
                TRADING_DAYS_PER_YEAR
            )
        )
    else:
        sortino = np.nan

    # ========================================================
    # MAXIMUM DRAWDOWN
    # ========================================================

    max_drawdown = (
        daily["Drawdown"]
        .min()
    )

    # ========================================================
    # CUMULATIVE RETURN
    # ========================================================

    cumulative_return = (
        daily["Equity"]
        .iloc[-1]
        - 1.0
    )

    # ========================================================
    # CAGR
    # ========================================================

    first_date = (
        daily["Event_Session"]
        .min()
    )

    last_date = (
        daily["Event_Session"]
        .max()
    )

    calendar_days = (
        last_date
        - first_date
    ).days

    years = (
        calendar_days
        / 365.25
    )

    final_equity = (
        daily["Equity"]
        .iloc[-1]
    )

    if (
        years > 0
        and final_equity > 0
    ):
        cagr = (
            final_equity
            ** (1 / years)
            - 1
        )
    else:
        cagr = np.nan

    return {
        "Num_Events":
            len(events),

        "Active_Sessions":
            int(
                (
                    daily["Num_Positions"] > 0
                ).sum()
            ),

        "EV":
            expected_value,

        "Median_Event_Return":
            event_returns.median(),

        "Win_Rate":
            win_rate,

        "Avg_Win":
            avg_win,

        "Avg_Loss":
            avg_loss,

        "Profit_Factor":
            profit_factor,

        "Sharpe":
            sharpe,

        "Sortino":
            sortino,

        "Max_Drawdown":
            max_drawdown,

        "Cumulative_Return":
            cumulative_return,

        "CAGR":
            cagr,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "FINANCIAL EVALUATION OF OOS PREDICTIONS"
    )

    print(
        "=" * 80
    )

    # ========================================================
    # WCZYTANIE
    # ========================================================

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    market = pd.read_csv(
        MARKET_PATH
    )

    validate_predictions(
        predictions
    )

    predictions[
        "Event_Session"
    ] = pd.to_datetime(
        predictions[
            "Event_Session"
        ]
    )

    market["Date"] = pd.to_datetime(
        market["Date"]
    )

    # ========================================================
    # KALENDARZ QQQ
    # ========================================================

    trading_calendar = (
        market.loc[
            market["Ticker"] == "QQQ",
            ["Date"],
        ]
        .drop_duplicates()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    print(
        f"\nLiczba predykcji OOS: "
        f"{len(predictions)}"
    )

    print(
        "\nModele:"
    )

    print(
        predictions[
            "Model"
        ]
        .value_counts()
    )

    # ========================================================
    # AGREGACJA FILINGÓW
    # ========================================================

    events = aggregate_filing_predictions(
        predictions
    )

    print(
        "\nLiczba decyzji "
        "Ticker + Event_Session:"
    )

    print(
        events.groupby(
            "Model"
        )
        .size()
    )

    # ========================================================
    # SESSION RETURNS
    # ========================================================

    session_returns = (
        build_session_returns(
            events
        )
    )

    # ========================================================
    # OVERALL
    # ========================================================

    overall_rows = []

    all_daily_rows = []

    for model_name in sorted(
        events["Model"].unique()
    ):

        model_events = events[
            events["Model"]
            == model_name
        ].copy()

        model_sessions = (
            session_returns[
                session_returns["Model"]
                == model_name
            ]
            .copy()
        )

        daily = build_daily_returns(
            model_sessions=model_sessions,
            trading_calendar=trading_calendar,
        )

        daily["Model"] = model_name

        all_daily_rows.append(
            daily
        )

        metrics = (
            calculate_financial_metrics(
                events=model_events,
                daily=daily,
            )
        )

        overall_rows.append(
            {
                "Model":
                    model_name,
                **metrics,
            }
        )

    overall_df = pd.DataFrame(
        overall_rows
    )

    # ========================================================
    # PER YEAR
    # ========================================================

    yearly_rows = []

    for (
        model_name,
        test_year,
    ), year_events in events.groupby(
        [
            "Model",
            "Test_Year",
        ]
    ):

        year_sessions = (
            session_returns[
                (
                    session_returns[
                        "Model"
                    ] == model_name
                )
                & (
                    session_returns[
                        "Test_Year"
                    ] == test_year
                )
            ]
            .copy()
        )

        daily = build_daily_returns(
            model_sessions=year_sessions,
            trading_calendar=trading_calendar,
        )

        metrics = (
            calculate_financial_metrics(
                events=year_events,
                daily=daily,
            )
        )

        yearly_rows.append(
            {
                "Model":
                    model_name,

                "Test_Year":
                    int(test_year),

                **metrics,
            }
        )

    yearly_df = (
        pd.DataFrame(
            yearly_rows
        )
        .sort_values(
            [
                "Test_Year",
                "Model",
            ]
        )
    )

    # ========================================================
    # ZAPIS
    # ========================================================

    daily_df = pd.concat(
        all_daily_rows,
        ignore_index=True,
    )

    overall_df.to_csv(
        OUTPUT_OVERALL,
        index=False,
    )

    yearly_df.to_csv(
        OUTPUT_YEARLY,
        index=False,
    )

    events.to_csv(
        OUTPUT_EVENTS,
        index=False,
    )

    daily_df.to_csv(
        OUTPUT_DAILY,
        index=False,
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    display_columns = [
        "Model",
        "Num_Events",
        "EV",
        "Win_Rate",
        "Profit_Factor",
        "Sharpe",
        "Sortino",
        "Max_Drawdown",
        "Cumulative_Return",
        "CAGR",
    ]

    print(
        "\n"
        + "=" * 80
    )

    print(
        "WYNIKI FINANSOWE - CAŁY OOS"
    )

    print(
        "=" * 80
    )

    print(
        overall_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "WYNIKI FINANSOWE - 2026"
    )

    print(
        "=" * 80
    )

    results_2026 = (
        yearly_df[
            yearly_df[
                "Test_Year"
            ] == 2026
        ]
    )

    print(
        results_2026[
            [
                "Model",
                "Num_Events",
                "EV",
                "Win_Rate",
                "Profit_Factor",
                "Sharpe",
                "Sortino",
                "Max_Drawdown",
                "Cumulative_Return",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nPliki zapisano:"
    )

    print(
        OUTPUT_OVERALL
    )

    print(
        OUTPUT_YEARLY
    )

    print(
        OUTPUT_EVENTS
    )

    print(
        OUTPUT_DAILY
    )


if __name__ == "__main__":
    main()