# run_single_asset.py

from config.settings import TRAIN_START_DATE, TRAIN_END_DATE, TRADE_START_DATE, TRADE_END_DATE, SINGLE_ASSET_TICKERS, INDICATORS
from data.data_fetcher import fetch_data
from data.feature_engineer import FeatureEngineer, add_cov_and_returns
from models.trainer import train_single_asset_agent
from models.evaluator import evaluate_agent
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
import os

os.makedirs("results", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

def get_env(df, risk_aware=False):
    stock_dim = 1
    state_space = 1 + 2 * stock_dim + len(INDICATORS) * stock_dim
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1_000_000,
        "num_stock_shares": [0],
        "buy_cost_pct": [0.001],
        "sell_cost_pct": [0.001],
        "state_space": state_space,
        "stock_dim": stock_dim,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dim,
        "reward_scaling": 1e-4
    }
    return StockTradingEnv(df=df, **env_kwargs)

def main():
    fe = FeatureEngineer(
        use_technical_indicator=True,
        use_turbulence=False,
        user_defined_feature=False
    )

    for symbol in SINGLE_ASSET_TICKERS:
        print(f"📥 Processing {symbol}...")
        df = fetch_data([symbol], TRAIN_START_DATE, TRADE_END_DATE)
        df = fe.preprocess_data(df)
        df = add_cov_and_returns(df)
        df = df.fillna(0)

        train_df = data_split(df, TRAIN_START_DATE, TRAIN_END_DATE)
        test_df = data_split(df, TRADE_START_DATE, TRADE_END_DATE)

        trained_models = train_single_asset_agent(train_df, get_env_fn=get_env, symbol=symbol, risk_aware=False)

        for algo, model in trained_models.items():
            print(f"📈 Evaluating {symbol} - {algo.upper()}")
            test_env = get_env(test_df, risk_aware=False)
            evaluate_agent(model, test_env, label=f"{symbol}_{algo}", save_path=f"results/{symbol}_{algo}_curve.png")

if __name__ == "__main__":
    main()
