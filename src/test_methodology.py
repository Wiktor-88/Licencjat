# Krótkie testy najważniejszych założeń metodologicznych

import unittest

import numpy as np
import pandas as pd

from src.features.aggregate_sentiment import aggregate_single_filing
from src.features.build_model_dataset import add_sentiment_momentum, add_signal_timing, add_tradable_target
from src.models.evaluate_financial_metrics import (apply_strategy, attach_execution_data,
    calculate_relative_momentum, expand_portfolio_events)
from src.models.metrics_permutations_test import permute_targets_by_session
from src.models.model_utils import aggregate_model_events
from src.models.stat_test_utils import run_paired_permutation_test


class MethodologyTest(unittest.TestCase):
    def test_sentiment_uses_token_weights_and_detects_events(self) -> None:
        blocks = pd.DataFrame([
            {"Ticker": "AAPL", "Date": "2025-01-01", "Accession": "1", "Source_Type": "EX-99.1", "Item_Number": "N/A",
                "Block_ID": 0, "Text_Snippet": "Quarterly earnings and guidance.", "Token_Count": 10,
                "Predicted_Label": "Positive", "Prob_Positive": 0.8, "Prob_Negative": 0.2, "Prob_Neutral": 0.0, "Net_Sentiment": 0.6},
            {"Ticker": "AAPL", "Date": "2025-01-01", "Accession": "1", "Source_Type": "EX-99.1", "Item_Number": "N/A",
                "Block_ID": 1, "Text_Snippet": "A share repurchase program.", "Token_Count": 30,
                "Predicted_Label": "Negative", "Prob_Positive": 0.2, "Prob_Negative": 0.8, "Prob_Neutral": 0.0, "Net_Sentiment": -0.6}])

        result = aggregate_single_filing(blocks)
        self.assertAlmostEqual(result["Mean_Positive"], 0.35)
        self.assertAlmostEqual(result["Mean_Net_Sentiment"], -0.3)
        self.assertEqual(result["Has_Event_Earnings"], 1)
        self.assertEqual(result["Has_Event_Guidance"], 1)
        self.assertEqual(result["Has_Event_Buyback"], 1)

    def test_cluster_permutation_keeps_session_vectors(self) -> None:
        frame = pd.DataFrame({"Test_Year": [2025] * 4, "Event_Session": ["2025-01-02"] * 2 + ["2025-01-03"] * 2,
            "y_true": [0, 1, 1, 1]})
        permuted, movable = permute_targets_by_session(frame, np.random.default_rng(4))
        vectors = {tuple(permuted[:2]), tuple(permuted[2:])}
        self.assertEqual(vectors, {(0, 1), (1, 1)})
        self.assertEqual(movable, 2)

    def test_paired_test_reports_session_count(self) -> None:
        pair = pd.DataFrame({"Ticker": ["A", "B", "A", "B"], "Event_Session": ["2025-01-02"] * 2 + ["2025-01-03"] * 2,
            "Accession": ["1", "2", "3", "4"], "Test_Year": [2025] * 4, "y_true": [0, 1, 0, 1],
            "y_pred_A": [0, 0, 0, 0], "y_pred_B": [0, 1, 0, 1], "y_prob_A": [0.4, 0.4, 0.4, 0.4],
            "y_prob_B": [0.2, 0.8, 0.2, 0.8]})
        result = run_paired_permutation_test(pair, n_permutations=20, random_state=3)
        self.assertEqual(result["N_Sessions"], 2)
        self.assertGreater(result["Delta_BA"], 0)

    def test_financial_cost_is_round_trip(self) -> None:
        events = pd.DataFrame({"y_prob": [0.8], "Market_Neutral_Return": [0.02]})
        result = apply_strategy(events, cost_bps=5, margin=0)
        self.assertAlmostEqual(result.loc[0, "Transaction_Cost"], 0.001)
        self.assertAlmostEqual(result.loc[0, "Strategy_Return_Net"], 0.019)

    def test_tradable_target_uses_event_open_and_close(self) -> None:
        frame = pd.DataFrame({"Event_Open": [100.0], "Event_Close": [110.0],
                              "QQQ_Event_Open": [100.0], "QQQ_Event_Close": [101.0]})
        result = add_tradable_target(frame)
        self.assertAlmostEqual(result.loc[0, "Tradable_Abnormal_Return_1D"], 0.09)
        self.assertEqual(result.loc[0, "Target_Tradable_Abnormal_1D"], 1)

    def test_signal_must_be_known_fifteen_minutes_before_open(self) -> None:
        frame = pd.DataFrame({"Event_Session": ["2025-01-02", "2025-01-02"],
                              "Acceptance_DateTime_ET": ["2025-01-02T08:00:00-05:00",
                                                         "2025-01-02T09:20:00-05:00"]})
        result = add_signal_timing(frame)
        self.assertEqual(result["Signal_Available_Before_Open"].tolist(), [1, 0])
        self.assertAlmostEqual(result.loc[0, "Signal_Lead_Minutes"], 90.0)
        self.assertAlmostEqual(result.loc[1, "Signal_Lead_Minutes"], 10.0)

    def test_multiple_filings_create_one_model_event(self) -> None:
        frame = pd.DataFrame([
            {"Ticker": "AAPL", "Event_Session": "2025-01-02", "Accession": "1",
             "Acceptance_DateTime_ET": "2025-01-01T16:10:00-05:00", "Feature_Cutoff_Session": "2024-12-31",
             "Target_Tradable_Abnormal_1D": 1, "Tradable_Abnormal_Return_1D": 0.02,
             "Mean_Net_Sentiment": 0.2, "Sentiment_Total_Tokens": 10, "Has_Event_Earnings": 1},
            {"Ticker": "AAPL", "Event_Session": "2025-01-02", "Accession": "2",
             "Acceptance_DateTime_ET": "2025-01-01T17:00:00-05:00", "Feature_Cutoff_Session": "2024-12-31",
             "Target_Tradable_Abnormal_1D": 1, "Tradable_Abnormal_Return_1D": 0.02,
             "Mean_Net_Sentiment": -0.2, "Sentiment_Total_Tokens": 30, "Has_Event_Earnings": 0}])
        result = aggregate_model_events(frame, "Target_Tradable_Abnormal_1D")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "Event_Filing_Count"], 2)
        self.assertEqual(result.loc[0, "Has_Event_Earnings"], 1)
        self.assertAlmostEqual(result.loc[0, "Mean_Net_Sentiment"], -0.1)

    def test_sentiment_history_uses_previous_sessions(self) -> None:
        frame = pd.DataFrame({"Ticker": ["AAPL"] * 3,
                              "Event_Session": pd.to_datetime(["2025-01-02", "2025-01-02", "2025-01-03"]),
                              "Mean_Net_Sentiment": [0.2, -0.2, 0.6],
                              "Sentiment_Total_Tokens": [10, 10, 10]})
        result = add_sentiment_momentum(frame)
        next_session = result[result["Event_Session"].eq(pd.Timestamp("2025-01-03"))].iloc[0]
        self.assertAlmostEqual(next_session["Previous_Sentiment_Mean_3"], 0.0)
        self.assertAlmostEqual(next_session["Sentiment_Momentum_3"], 0.6)

    def test_financial_return_matches_model_target(self) -> None:
        predictions = pd.DataFrame({"Family": ["Test"], "Model": ["Model A"], "Ticker": ["AAPL"],
                                    "Accession": ["1"], "Event_Session": pd.to_datetime(["2025-01-02"]),
                                    "Test_Year": [2025], "y_true": [1], "y_prob": [0.8],
                                    "Tradable_Abnormal_Return_1D": [0.09]})
        metadata = pd.DataFrame({"Ticker": ["AAPL"], "Accession": ["1"],
                                 "Event_Session": pd.to_datetime(["2025-01-02"]),
                                 "Acceptance_DateTime_ET": pd.to_datetime(["2025-01-02T08:00:00-05:00"], utc=True),
                                 "Publication_Period": ["PRE_MARKET"],
                                 "Feature_Cutoff_Session": pd.to_datetime(["2025-01-01"])})
        market = pd.DataFrame({"Ticker": ["AAPL", "QQQ"],
                               "Date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
                               "Open": [100.0, 100.0], "Close": [110.0, 101.0], "Adj_Close": [110.0, 101.0]})
        eligible, excluded = attach_execution_data(predictions, metadata, market)
        self.assertTrue(excluded.empty)
        self.assertAlmostEqual(eligible.loc[0, "Market_Neutral_Return"], 0.045)

    def test_momentum_uses_only_prices_up_to_cutoff(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=62)
        market = pd.concat([
            pd.DataFrame({"Ticker": "AAPL", "Date": dates, "Adj_Close": np.arange(100.0, 162.0)}),
            pd.DataFrame({"Ticker": "QQQ", "Date": dates, "Adj_Close": np.arange(100.0, 131.0, 0.5)})],
            ignore_index=True)
        momentum = calculate_relative_momentum(market)
        last = momentum.loc[momentum["Ticker"].eq("AAPL")].iloc[-1]
        expected = (161 / 101 - 1) - (130.5 / 100.5 - 1)
        self.assertAlmostEqual(last["Momentum_Score_60D"], expected)

    def test_four_portfolio_types_are_created(self) -> None:
        events = pd.DataFrame({"Ticker": ["INTC", "NVDA", "AMD"],
                               "Momentum_Score_60D": [0.1, -0.1, 0.2]})
        result = expand_portfolio_events(events)
        self.assertEqual(set(result["Portfolio_Type"]), {"Dividend", "Value", "Growth", "Momentum"})
        self.assertEqual(set(result.loc[result["Portfolio_Type"].eq("Value"), "Ticker"]), {"INTC"})
        self.assertEqual(set(result.loc[result["Portfolio_Type"].eq("Momentum"), "Ticker"]), {"INTC", "AMD"})


if __name__ == "__main__":
    unittest.main()
