# Wspólne ustawienia i listy cech dla modeli

TARGET = "Target_Tradable_Abnormal_1D"
TEST_YEARS = [2023, 2024, 2025, 2026]
MIN_SEC_COUNT = 5
RANDOM_STATE = 67

CATEGORICAL_FEATURES = ["Ticker"]

MARKET_FEATURES = [
    "Stock_vs_QQQ_1D",
    "Stock_vs_QQQ_3D",
    "Stock_vs_QQQ_5D",
    "Volatility_14D",
    "Relative_Volume_20D",
    "RSI_14",
    "Price_to_SMA20",
    "Intraday_Return",
    "Daily_Range",
    "QQQ_Log_Return_1D",
    "QQQ_Log_Return_3D",
    "QQQ_Log_Return_5D",
    "QQQ_Volatility_14D",
]

MARKET_Z_FEATURES = [
    "Log_Return_1D_Z60",
    "Log_Return_3D_Z60",
    "Log_Return_5D_Z60",
    "Volatility_14D_Z60",
    "Relative_Volume_20D_Z60",
    "RSI_14",
    "Price_to_SMA20_Z60",
    "Intraday_Return_Z60",
    "Daily_Range_Z60",
    "QQQ_Log_Return_1D",
    "QQQ_Log_Return_3D",
    "QQQ_Log_Return_5D",
    "QQQ_Volatility_14D",
]

MARKET_COMPACT_FEATURES = [
    "Log_Return_1D_Z60",
    "Log_Return_5D_Z60",
    "Volatility_14D_Z60",
    "Relative_Volume_20D_Z60",
    "RSI_14",
    "Price_to_SMA20_Z60",
    "Intraday_Return_Z60",
    "Daily_Range_Z60",
    "QQQ_Log_Return_1D",
    "QQQ_Log_Return_5D",
    "QQQ_Volatility_14D",
]

SEC_BINARY_CANDIDATES = [
    "Has_EX99",
    "Has_Event_Earnings",
    "Has_Event_Guidance",
    "Has_Event_Buyback",
    "Has_Item_1_01",
    "Has_Item_1_02",
    "Has_Item_1_05",
    "Has_Item_2_01",
    "Has_Item_2_02",
    "Has_Item_2_03",
    "Has_Item_5_02",
    "Has_Item_5_03",
    "Has_Item_5_07",
    "Has_Item_7_01",
    "Has_Item_8_01",
]

SENTIMENT_FEATURES = [
    "Mean_Net_Sentiment",
    "Sentiment_Momentum_3",
]

SENTIMENT_HISTORY_FLAG = "Has_Sentiment_History"

SENTIMENT_CONTEXT_FEATURES = [
    "Mean_Net_Sentiment",
    "Sentiment_Momentum_3",
    "Abs_Sentiment",
    "Sentiment_x_Prior_Return_5D",
]

# Ten sam zestaw wejściowy pozwala uczciwie porównywać modele tabularne.
TABULAR_MARKET_FEATURES = MARKET_FEATURES
