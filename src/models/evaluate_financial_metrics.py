"""Ocena ekonomiczna zapisanych predykcji out-of-sample.

Skrypt nie trenuje modeli. Sprawdza, czy sygnały znane przed otwarciem sesji
miały wartość po kosztach transakcyjnych, także w czterech przekrojach:
dywidendowym, wartościowym, wzrostowym i momentum. Pozycja jest utrzymywana od
otwarcia do zamknięcia sesji zdarzenia, a ekspozycja rynkowa jest ograniczana
przez przeciwną pozycję w QQQ.
"""

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from scipy import stats

from src.models.stat_test_utils import holm_adjust


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DATASET_PATH = DATA_DIR / "model_dataset.csv"
MARKET_PATH = DATA_DIR / "market_features.csv"
OUTPUT_OVERALL = DATA_DIR / "financial_metrics_overall.csv"
OUTPUT_YEARLY = DATA_DIR / "financial_metrics_by_year.csv"
OUTPUT_EVENTS = DATA_DIR / "financial_event_returns.csv"
OUTPUT_DAILY = DATA_DIR / "financial_daily_returns.csv"
OUTPUT_SENSITIVITY = DATA_DIR / "financial_metrics_sensitivity.csv"
OUTPUT_EXCLUDED = DATA_DIR / "financial_excluded_events.csv"
OUTPUT_PORTFOLIO_OVERALL = DATA_DIR / "portfolio_financial_metrics_overall.csv"
OUTPUT_PORTFOLIO_YEARLY = DATA_DIR / "portfolio_financial_metrics_by_year.csv"
OUTPUT_PORTFOLIO_EVENTS = DATA_DIR / "portfolio_financial_event_returns.csv"
OUTPUT_PORTFOLIO_DAILY = DATA_DIR / "portfolio_financial_daily_returns.csv"
OUTPUT_PORTFOLIO_SENSITIVITY = DATA_DIR / "portfolio_financial_metrics_sensitivity.csv"

PREDICTION_FILES = {
    "Logistic": DATA_DIR / "oos_predictions.csv",
    "Tree": DATA_DIR / "decision_tree_oos_predictions.csv",
    "XGBoost": DATA_DIR / "xgboost_oos_predictions.csv",
    "CatBoost": DATA_DIR / "catboost_oos_predictions.csv",
    "TabNet": DATA_DIR / "tabnet_oos_predictions.csv",
    "LSTM": DATA_DIR / "lstm_oos_predictions.csv",
    "Transformer": DATA_DIR / "transformer_oos_predictions.csv"}

CANONICAL_MODELS = {
    "Logistic": ["MODEL A - MARKET", "MODEL B - MARKET + SEC", "MODEL C - MARKET + SEC + FINBERT"],
    "Tree": ["TREE A - MARKET", "TREE B - MARKET + SEC", "TREE C - MARKET + SEC + FINBERT"],
    "XGBoost": ["XGB A - MARKET", "XGB B - MARKET + SEC", "XGB C - MARKET + SEC + FINBERT"],
    "CatBoost": ["CAT A - MARKET", "CAT B - MARKET + SEC", "CAT C - MARKET + SEC + FINBERT"],
    "TabNet": ["TABNET A - MARKET", "TABNET B - MARKET + SEC", "TABNET C - MARKET + SEC + FINBERT"],
    "LSTM": ["LSTM A - MARKET SEQUENCE", "LSTM B - MARKET SEQUENCE + SEC", "LSTM C - MARKET SEQUENCE + SEC + FINBERT"],
    "Transformer": ["TRANSFORMER A - MARKET SEQUENCE", "TRANSFORMER B - MARKET SEQUENCE + SEC", "TRANSFORMER C - MARKET SEQUENCE + SEC + FINBERT"]}

PRIMARY_COST_BPS = 5.0
PRIMARY_MARGIN = 0.0
COST_SCENARIOS_BPS = [0.0, 5.0, 10.0]
MARGIN_SCENARIOS = [0.0, 0.05, 0.10]
MIN_LEAD_MINUTES = 15
MARKET_CALENDAR_NAME = "NASDAQ"
TRADING_DAYS_PER_YEAR = 252
HAC_MAX_LAGS = 5
MOMENTUM_LOOKBACK = 60

# To są stałe, eksploracyjne podzbiory badanej próby, a nie replika indeksów faktorowych.
STATIC_PORTFOLIO_TICKERS = {
    "Dividend": {"AAPL", "MSFT", "NVDA", "INTC", "AVGO", "QCOM"},
    "Value": {"INTC", "QCOM"},
    "Growth": {"NVDA", "AMD", "AMZN", "GOOGL", "META", "TSLA", "NFLX", "ADBE"}}
PORTFOLIO_DEFINITIONS = {
    "Dividend": "fixed_ex_ante_dividend_payers",
    "Value": "fixed_ex_ante_mature_semiconductor_proxy",
    "Growth": "fixed_ex_ante_growth_companies",
    "Momentum": f"positive_relative_momentum_{MOMENTUM_LOOKBACK}d"}

logger = logging.getLogger(__name__)


def validate_columns(df: pd.DataFrame, required: list[str], source: Path) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Brakuje kolumn w {source.name}: {', '.join(missing)}")


def load_predictions() -> pd.DataFrame:
    frames = []
    required = ["Ticker", "Event_Session", "Accession", "Test_Year", "Model", "y_true", "y_pred", "y_prob",
                "Tradable_Abnormal_Return_1D"]

    for family, path in PREDICTION_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Brak pliku predykcji dla {family}: {path}")

        frame = pd.read_csv(path)
        validate_columns(frame, required, path)
        frame = frame.loc[frame["Model"].isin(CANONICAL_MODELS[family])].copy()
        found_models = set(frame["Model"].unique())
        if found_models != set(CANONICAL_MODELS[family]):
            raise ValueError(f"{family}: oczekiwano trzech modeli A/B/C, znaleziono {sorted(found_models)}")
        frame["Family"] = family
        frames.append(frame)

    if not frames:
        raise FileNotFoundError("Nie znaleziono żadnych plików z predykcjami OOS.")

    predictions = pd.concat(frames, ignore_index=True)
    predictions["Event_Session"] = pd.to_datetime(predictions["Event_Session"], errors="raise").dt.normalize()
    predictions["Test_Year"] = pd.to_numeric(predictions["Test_Year"], errors="raise").astype(int)
    predictions["y_prob"] = pd.to_numeric(predictions["y_prob"], errors="raise")
    predictions["Tradable_Abnormal_Return_1D"] = pd.to_numeric(
        predictions["Tradable_Abnormal_Return_1D"], errors="raise")

    invalid_prob = predictions["y_prob"].isna() | ~predictions["y_prob"].between(0, 1)
    if invalid_prob.any():
        raise ValueError("Kolumna y_prob zawiera braki lub wartości spoza przedziału [0, 1].")
    if predictions["Test_Year"].ne(predictions["Event_Session"].dt.year).any():
        raise ValueError("Test_Year nie zgadza się z rokiem Event_Session.")

    key = ["Family", "Model", "Ticker", "Accession", "Event_Session"]
    if predictions.duplicated(key).any():
        raise ValueError("Predykcje OOS zawierają zduplikowane obserwacje tego samego modelu i filingu.")

    logger.info("Wczytano %d predykcji OOS dla %d modeli.", len(predictions), predictions["Model"].nunique())
    return predictions


def load_event_metadata() -> pd.DataFrame:
    required = ["Ticker", "Accession", "Event_Session", "Acceptance_DateTime_ET", "Publication_Period",
                "Feature_Cutoff_Session"]
    metadata = pd.read_csv(MODEL_DATASET_PATH, usecols=lambda column: column in required)
    validate_columns(metadata, required, MODEL_DATASET_PATH)
    metadata["Event_Session"] = pd.to_datetime(metadata["Event_Session"], errors="raise").dt.normalize()
    metadata["Feature_Cutoff_Session"] = pd.to_datetime(metadata["Feature_Cutoff_Session"], errors="raise").dt.normalize()
    metadata["Acceptance_DateTime_ET"] = pd.to_datetime(metadata["Acceptance_DateTime_ET"], errors="coerce", utc=True)
    return metadata.drop_duplicates(["Ticker", "Accession", "Event_Session"])


def load_market() -> pd.DataFrame:
    required = ["Ticker", "Date", "Open", "Close", "Adj_Close"]
    market = pd.read_csv(MARKET_PATH, usecols=lambda column: column in required)
    validate_columns(market, required, MARKET_PATH)
    market["Date"] = pd.to_datetime(market["Date"], errors="raise").dt.normalize()
    for column in ["Open", "Close", "Adj_Close"]:
        market[column] = pd.to_numeric(market[column], errors="coerce")
    return market.drop_duplicates(["Ticker", "Date"])


def calculate_relative_momentum(market: pd.DataFrame) -> pd.DataFrame:
    prices = market[["Ticker", "Date", "Adj_Close"]].sort_values(["Ticker", "Date"]).copy()
    prices["Return_60D"] = prices.groupby("Ticker")["Adj_Close"].pct_change(MOMENTUM_LOOKBACK, fill_method=None)
    qqq = prices.loc[prices["Ticker"].eq("QQQ"), ["Date", "Return_60D"]].rename(
        columns={"Return_60D": "QQQ_Return_60D"})
    momentum = prices.loc[prices["Ticker"].ne("QQQ")].merge(qqq, on="Date", how="left", validate="many_to_one")
    momentum["Momentum_Score_60D"] = momentum["Return_60D"] - momentum["QQQ_Return_60D"]
    return momentum.rename(columns={"Date": "Feature_Cutoff_Session"})[
        ["Ticker", "Feature_Cutoff_Session", "Momentum_Score_60D"]]


@lru_cache(maxsize=16)
def create_schedule(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    schedule = mcal.get_calendar(MARKET_CALENDAR_NAME).schedule(start_date=start, end_date=end).reset_index()
    schedule = schedule.rename(columns={"index": "Event_Session", "market_open": "Market_Open_UTC"})
    schedule["Event_Session"] = pd.to_datetime(schedule["Event_Session"]).dt.tz_localize(None).dt.normalize()
    schedule["Market_Open_UTC"] = pd.to_datetime(schedule["Market_Open_UTC"], utc=True)
    return schedule[["Event_Session", "Market_Open_UTC"]]


def attach_execution_data(predictions: pd.DataFrame, metadata: pd.DataFrame, market: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    start, end = predictions["Event_Session"].min(), predictions["Event_Session"].max()
    schedule = create_schedule(start, end)
    stock = market.rename(columns={"Date": "Event_Session", "Open": "Stock_Open", "Close": "Stock_Close"})
    qqq = market.loc[market["Ticker"].eq("QQQ"), ["Date", "Open", "Close"]].rename(
        columns={"Date": "Event_Session", "Open": "QQQ_Open", "Close": "QQQ_Close"})

    momentum = calculate_relative_momentum(market)
    events = predictions.merge(metadata, on=["Ticker", "Accession", "Event_Session"], how="left", validate="many_to_one")
    events = events.merge(momentum, on=["Ticker", "Feature_Cutoff_Session"], how="left", validate="many_to_one")
    events = events.merge(schedule, on="Event_Session", how="left", validate="many_to_one")
    events = events.merge(stock[["Ticker", "Event_Session", "Stock_Open", "Stock_Close"]],
        on=["Ticker", "Event_Session"], how="left", validate="many_to_one")
    events = events.merge(qqq, on="Event_Session", how="left", validate="many_to_one")
    events["Lead_Minutes"] = (events["Market_Open_UTC"] - events["Acceptance_DateTime_ET"]).dt.total_seconds() / 60

    missing_metadata = events["Acceptance_DateTime_ET"].isna() | events["Market_Open_UTC"].isna()
    missing_prices = events[["Stock_Open", "Stock_Close", "QQQ_Open", "QQQ_Close"]].isna().any(axis=1)
    invalid_prices = events[["Stock_Open", "QQQ_Open"]].le(0).any(axis=1)
    late_signal = events["Lead_Minutes"].lt(MIN_LEAD_MINUTES)
    events["Exclusion_Reason"] = np.select(
        [missing_metadata, missing_prices | invalid_prices, late_signal],
        ["missing_event_metadata", "missing_execution_prices", "signal_known_too_late"], default="")

    excluded = events.loc[events["Exclusion_Reason"].ne("")].copy()
    eligible = events.loc[events["Exclusion_Reason"].eq("")].copy()
    eligible["Stock_Return_Open_Close"] = eligible["Stock_Close"] / eligible["Stock_Open"] - 1
    eligible["QQQ_Return_Open_Close"] = eligible["QQQ_Close"] / eligible["QQQ_Open"] - 1
    eligible["Calculated_Tradable_Abnormal_Return_1D"] = (
        eligible["Stock_Return_Open_Close"] - eligible["QQQ_Return_Open_Close"])

    return_matches = np.isclose(eligible["Tradable_Abnormal_Return_1D"],
                                eligible["Calculated_Tradable_Abnormal_Return_1D"],
                                rtol=1e-9, atol=1e-12)
    if not return_matches.all():
        raise ValueError("Stopa zwrotu w predykcjach nie zgadza się z cenami Open/Close.")
    expected_target = eligible["Calculated_Tradable_Abnormal_Return_1D"].gt(0).astype(int)
    if expected_target.ne(eligible["y_true"].astype(int)).any():
        raise ValueError("y_true nie zgadza się ze znakiem możliwego do zrealizowania zwrotu.")

    eligible["Market_Neutral_Return"] = 0.5 * (eligible["Stock_Return_Open_Close"] - eligible["QQQ_Return_Open_Close"])

    return_check = eligible.groupby(["Family", "Model", "Ticker", "Event_Session"])["Market_Neutral_Return"].nunique()
    if return_check.gt(1).any():
        raise ValueError("Niespójne stopy zwrotu dla tego samego tickera i sesji.")

    eligible = eligible.groupby(["Family", "Model", "Ticker", "Event_Session"], as_index=False).agg(
        Test_Year=("Test_Year", "first"), Filing_Count=("Accession", "size"), Accessions=("Accession", lambda values: " | ".join(values.astype(str))),
        y_prob=("y_prob", "mean"), Acceptance_DateTime_ET=("Acceptance_DateTime_ET", "max"), Lead_Minutes=("Lead_Minutes", "min"),
        Feature_Cutoff_Session=("Feature_Cutoff_Session", "first"), Momentum_Score_60D=("Momentum_Score_60D", "first"),
        Publication_Period=("Publication_Period", lambda values: " | ".join(sorted(set(values.astype(str))))),
        Stock_Return_Open_Close=("Stock_Return_Open_Close", "first"), QQQ_Return_Open_Close=("QQQ_Return_Open_Close", "first"),
        Market_Neutral_Return=("Market_Neutral_Return", "first"))

    logger.info("Do oceny dopuszczono %d decyzji; wykluczono %d wierszy predykcji.", len(eligible), len(excluded))
    return eligible, excluded


def apply_strategy(events: pd.DataFrame, cost_bps: float, margin: float) -> pd.DataFrame:
    result = events.copy()
    result["Active"] = result["y_prob"].sub(0.5).abs().ge(margin)
    result["Signal"] = np.where(result["y_prob"].ge(0.5), 1, -1)
    result["Strategy_Return_Gross"] = np.where(result["Active"], result["Signal"] * result["Market_Neutral_Return"], 0.0)
    result["Transaction_Cost"] = np.where(result["Active"], 2 * cost_bps / 10000, 0.0)
    result["Strategy_Return_Net"] = result["Strategy_Return_Gross"] - result["Transaction_Cost"]
    result["Win"] = result["Active"] & result["Strategy_Return_Net"].gt(0)
    result["Cost_Bps_Per_Side"] = cost_bps
    result["Probability_Margin"] = margin
    return result


def expand_portfolio_events(events: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for portfolio, tickers in STATIC_PORTFOLIO_TICKERS.items():
        frame = events.loc[events["Ticker"].isin(tickers)].copy()
        frame["Portfolio_Type"] = portfolio
        frame["Portfolio_Definition"] = PORTFOLIO_DEFINITIONS[portfolio]
        frames.append(frame)

    momentum = events.loc[events["Momentum_Score_60D"].gt(0)].copy()
    momentum["Portfolio_Type"] = "Momentum"
    momentum["Portfolio_Definition"] = PORTFOLIO_DEFINITIONS["Momentum"]
    frames.append(momentum)
    result = pd.concat(frames, ignore_index=True)

    missing = set(PORTFOLIO_DEFINITIONS) - set(result["Portfolio_Type"].unique())
    if missing:
        raise ValueError(f"Brak obserwacji dla portfeli: {', '.join(sorted(missing))}")
    return result


def build_daily_returns(events: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    active = events.loc[events["Active"]]
    sessions = active.groupby("Event_Session", as_index=False).agg(
        Strategy_Return_Gross=("Strategy_Return_Gross", "mean"), Strategy_Return_Net=("Strategy_Return_Net", "mean"),
        Num_Positions=("Ticker", "nunique"))
    daily = schedule[["Event_Session"]].merge(sessions, on="Event_Session", how="left")
    daily[["Strategy_Return_Gross", "Strategy_Return_Net"]] = daily[["Strategy_Return_Gross", "Strategy_Return_Net"]].fillna(0.0)
    daily["Num_Positions"] = daily["Num_Positions"].fillna(0).astype(int)
    daily["Equity"] = (1 + daily["Strategy_Return_Net"]).cumprod()
    daily["Running_Max"] = daily["Equity"].cummax()
    daily["Drawdown"] = daily["Equity"] / daily["Running_Max"] - 1
    return daily


def hac_mean_test(returns: pd.Series, max_lags: int = HAC_MAX_LAGS) -> tuple[float, float]:
    values = returns.dropna().to_numpy(dtype=float)
    if len(values) < 3:
        return np.nan, np.nan

    centered = values - values.mean()
    effective_lags = min(max_lags, len(values) - 1)
    long_run_variance = np.dot(centered, centered) / len(values)
    for lag in range(1, effective_lags + 1):
        covariance = np.dot(centered[lag:], centered[:-lag]) / len(values)
        long_run_variance += 2 * (1 - lag / (effective_lags + 1)) * covariance

    standard_error = np.sqrt(max(long_run_variance, 0) / len(values))
    if standard_error == 0:
        return np.nan, np.nan

    statistic = values.mean() / standard_error
    p_value = 2 * stats.t.sf(abs(statistic), df=len(values) - 1)
    return float(statistic), float(p_value)


def calculate_financial_metrics(events: pd.DataFrame, daily: pd.DataFrame) -> dict:
    active = events.loc[events["Active"]]
    event_returns = active["Strategy_Return_Net"]
    daily_returns = daily["Strategy_Return_Net"]
    wins, losses = event_returns[event_returns.gt(0)], event_returns[event_returns.lt(0)]
    daily_std = daily_returns.std(ddof=1)
    downside_deviation = np.sqrt(np.mean(np.minimum(daily_returns.to_numpy(), 0) ** 2))
    hac_t, hac_p = hac_mean_test(daily_returns)
    hac_p_positive = float(stats.t.sf(hac_t, df=len(daily_returns) - 1)) if pd.notna(hac_t) else np.nan
    first_date, last_date = daily["Event_Session"].min(), daily["Event_Session"].max()
    years = (last_date - first_date).days / 365.25 if pd.notna(first_date) and pd.notna(last_date) else 0
    final_equity = daily["Equity"].iloc[-1] if len(daily) else np.nan
    gross_loss = abs(losses.sum())

    return {
        "Num_Candidate_Events": len(events), "Num_Trades": len(active), "Trade_Coverage": len(active) / len(events) if len(events) else np.nan,
        "Active_Sessions": int(daily["Num_Positions"].gt(0).sum()), "Mean_Trade_Return_Net": event_returns.mean(),
        "Median_Trade_Return_Net": event_returns.median(), "Win_Rate": event_returns.gt(0).mean() if len(event_returns) else np.nan,
        "Avg_Win": wins.mean() if len(wins) else np.nan, "Avg_Loss": abs(losses.mean()) if len(losses) else np.nan,
        "Profit_Factor": wins.sum() / gross_loss if gross_loss > 0 else np.nan, "Mean_Daily_Return_Net": daily_returns.mean(),
        "Annualized_Return_Arithmetic": daily_returns.mean() * TRADING_DAYS_PER_YEAR,
        "Annualized_Volatility": daily_std * np.sqrt(TRADING_DAYS_PER_YEAR) if pd.notna(daily_std) else np.nan,
        "Sharpe": daily_returns.mean() / daily_std * np.sqrt(TRADING_DAYS_PER_YEAR) if pd.notna(daily_std) and daily_std > 0 else np.nan,
        "Sortino": daily_returns.mean() / downside_deviation * np.sqrt(TRADING_DAYS_PER_YEAR) if downside_deviation > 0 else np.nan,
        "HAC_t": hac_t, "HAC_p_value": hac_p, "HAC_p_value_Positive": hac_p_positive,
        "Max_Drawdown": daily["Drawdown"].min() if len(daily) else np.nan,
        "Cumulative_Return": final_equity - 1 if pd.notna(final_equity) else np.nan,
        "CAGR": final_equity ** (1 / years) - 1 if years > 0 and final_equity > 0 else np.nan}


def summarize_groups(events: pd.DataFrame, data_end: pd.Timestamp, group_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_rows, yearly_rows, daily_frames = [], [], []

    for keys, group_events in events.groupby(group_columns):
        keys = keys if isinstance(keys, tuple) else (keys,)
        labels = dict(zip(group_columns, keys))
        first_year, last_year = group_events["Test_Year"].min(), group_events["Test_Year"].max()
        schedule_end = min(pd.Timestamp(f"{int(last_year)}-12-31"), data_end)
        schedule = create_schedule(pd.Timestamp(f"{int(first_year)}-01-01"), schedule_end)
        daily = build_daily_returns(group_events, schedule)
        for column, value in labels.items():
            daily[column] = value
        daily["Cost_Bps_Per_Side"] = group_events["Cost_Bps_Per_Side"].iloc[0]
        daily["Probability_Margin"] = group_events["Probability_Margin"].iloc[0]
        daily_frames.append(daily)
        overall_rows.append({**labels, "Cost_Bps_Per_Side": group_events["Cost_Bps_Per_Side"].iloc[0],
            "Probability_Margin": group_events["Probability_Margin"].iloc[0],
            **calculate_financial_metrics(group_events, daily)})

        for year, year_events in group_events.groupby("Test_Year"):
            year_end = min(pd.Timestamp(f"{int(year)}-12-31"), data_end)
            year_schedule = create_schedule(pd.Timestamp(f"{int(year)}-01-01"), year_end)
            year_daily = build_daily_returns(year_events, year_schedule)
            yearly_rows.append({**labels, "Test_Year": int(year),
                "Cost_Bps_Per_Side": group_events["Cost_Bps_Per_Side"].iloc[0],
                "Probability_Margin": group_events["Probability_Margin"].iloc[0],
                **calculate_financial_metrics(year_events, year_daily)})

    overall = pd.DataFrame(overall_rows).sort_values(group_columns)
    overall["HAC_p_value_Holm"] = holm_adjust(overall["HAC_p_value"].fillna(1.0).tolist())
    overall["HAC_p_value_Positive_Holm"] = holm_adjust(overall["HAC_p_value_Positive"].fillna(1.0).tolist())
    yearly = pd.DataFrame(yearly_rows).sort_values(["Test_Year", *group_columns])
    yearly["HAC_p_value_Holm"] = yearly.groupby("Test_Year")["HAC_p_value"].transform(
        lambda values: holm_adjust(values.fillna(1.0).tolist()))
    yearly["HAC_p_value_Positive_Holm"] = yearly.groupby("Test_Year")["HAC_p_value_Positive"].transform(
        lambda values: holm_adjust(values.fillna(1.0).tolist()))
    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    return overall, yearly, daily


def evaluate_scenario(base_events: pd.DataFrame, data_end: pd.Timestamp, cost_bps: float, margin: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = apply_strategy(base_events, cost_bps, margin)
    overall, yearly, daily = summarize_groups(events, data_end, ["Family", "Model"])
    return overall, yearly, events, daily


def evaluate_portfolio_scenario(base_events: pd.DataFrame, data_end: pd.Timestamp, cost_bps: float, margin: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = expand_portfolio_events(apply_strategy(base_events, cost_bps, margin))
    groups = ["Family", "Model", "Portfolio_Type", "Portfolio_Definition"]
    overall, yearly, daily = summarize_groups(events, data_end, groups)
    return overall, yearly, events, daily


def main() -> None:
    predictions = load_predictions()
    metadata = load_event_metadata()
    market = load_market()
    base_events, excluded = attach_execution_data(predictions, metadata, market)

    data_end = market.loc[market["Ticker"].eq("QQQ"), "Date"].max()
    overall, yearly, events, daily = evaluate_scenario(base_events, data_end, PRIMARY_COST_BPS, PRIMARY_MARGIN)
    portfolio_overall, portfolio_yearly, portfolio_events, portfolio_daily = evaluate_portfolio_scenario(
        base_events, data_end, PRIMARY_COST_BPS, PRIMARY_MARGIN)
    sensitivity_rows, portfolio_sensitivity_rows = [], []
    for cost_bps in COST_SCENARIOS_BPS:
        for margin in MARGIN_SCENARIOS:
            scenario, _, _, _ = evaluate_scenario(base_events, data_end, cost_bps, margin)
            portfolio_scenario, _, _, _ = evaluate_portfolio_scenario(base_events, data_end, cost_bps, margin)
            sensitivity_rows.append(scenario)
            portfolio_sensitivity_rows.append(portfolio_scenario)

    OUTPUT_OVERALL.parent.mkdir(parents=True, exist_ok=True)
    overall.to_csv(OUTPUT_OVERALL, index=False)
    yearly.to_csv(OUTPUT_YEARLY, index=False)
    events.to_csv(OUTPUT_EVENTS, index=False)
    daily.to_csv(OUTPUT_DAILY, index=False)
    sensitivity = pd.concat(sensitivity_rows, ignore_index=True)
    sensitivity["HAC_p_value_Positive_Global_Holm"] = holm_adjust(sensitivity["HAC_p_value_Positive"].fillna(1.0).tolist())
    sensitivity.to_csv(OUTPUT_SENSITIVITY, index=False)
    portfolio_overall.to_csv(OUTPUT_PORTFOLIO_OVERALL, index=False)
    portfolio_yearly.to_csv(OUTPUT_PORTFOLIO_YEARLY, index=False)
    portfolio_events.to_csv(OUTPUT_PORTFOLIO_EVENTS, index=False)
    portfolio_daily.to_csv(OUTPUT_PORTFOLIO_DAILY, index=False)
    portfolio_sensitivity = pd.concat(portfolio_sensitivity_rows, ignore_index=True)
    portfolio_sensitivity["HAC_p_value_Positive_Global_Holm"] = holm_adjust(
        portfolio_sensitivity["HAC_p_value_Positive"].fillna(1.0).tolist())
    portfolio_sensitivity.to_csv(OUTPUT_PORTFOLIO_SENSITIVITY, index=False)
    excluded.drop_duplicates(["Family", "Model", "Ticker", "Accession", "Event_Session"]).to_csv(OUTPUT_EXCLUDED, index=False)

    logger.info("Zapisano główną ocenę dla kosztu %.1f pb na stronę i marginesu %.2f.", PRIMARY_COST_BPS, PRIMARY_MARGIN)
    logger.info("Zapisano ocenę portfeli: %s", ", ".join(PORTFOLIO_DEFINITIONS))
    logger.info("Wyniki: %s", OUTPUT_OVERALL)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
