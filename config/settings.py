# config/settings.py

# Dates
TRAIN_START_DATE = "2011-01-01"
TRAIN_END_DATE = "2020-01-01"
TRADE_START_DATE = "2020-01-01"
TRADE_END_DATE = "2022-12-31"

# Single Asset Tickers
SINGLE_ASSET_TICKERS = ['AAPL', 'TSLA', 'JNJ']

# Multi-Asset Tickers
from finrl.config_tickers import DOW_30_TICKER
MULTI_ASSET_TICKERS = DOW_30_TICKER

# Technical indicators
INDICATORS = [
    'macd', 'rsi_30', 'cci_30', 'dx_30',
    'close_30_sma', 'close_60_sma'
]

MULTI_ASSET_FEATURES = ['macd', 'rsi_30', 'cci_30', 'dx_30']
