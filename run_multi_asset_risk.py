# run_multi_asset_risk.py

from config.settings import MULTI_ASSET_TICKERS, MULTI_ASSET_FEATURES, TRAIN_START_DATE, TRAIN_END_DATE, TRADE_START_DATE, TRADE_END_DATE
from data.data_fetcher import fetch_data
from data.feature_engineer import FeatureEngineer, add_cov_and_returns
from models.trainer import train_multi_asset_agent
from models.evaluator import evaluate_agent
from envs.multi_asset_env import MultiAssetStockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split
import torch
import os

os.makedirs("results", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

def get_env(df, stock_dim, risk_aware=True):
    state_space = stock_dim
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 10000,
        "transaction_cost_pct": 0,
        "state_space": state_space,
        "stock_dim": stock_dim,
        "tech_indicator_list": MULTI_ASSET_FEATURES,
        "action_space": stock_dim,
        "reward_scaling": 1
    }
    return MultiAssetStockTradingEnv(df=df, **env_kwargs)

def main():
    print("🔄 Loading multi-asset data with risk-aware reward...")
    df = fetch_data(MULTI_ASSET_TICKERS, TRAIN_START_DATE, TRADE_END_DATE)
    fe = FeatureEngineer(use_technical_indicator=True, use_turbulence=False, user_defined_feature=False)
    df = fe.preprocess_data(df)
    df = add_cov_and_returns(df)
    df = df.fillna(0)

    train = data_split(df, TRAIN_START_DATE, TRAIN_END_DATE)
    test = data_split(df, TRADE_START_DATE, TRADE_END_DATE)

    stock_dim = len(train.tic.unique())
    env_train = get_env(train, stock_dim, risk_aware=True)

    print("⚙️ Training risk-aware multi-asset agent...")
    model = train_multi_asset_agent(
        env_train,
        algo="sac",  # You can switch to "ddpg" if needed
        model_kwargs={
            "batch_size": 4096,
            "buffer_size": 10000,
            "learning_rate": 3e-4,
            "learning_starts": 100,
            "device": 'cuda' if torch.cuda.is_available() else 'cpu',
            "action_noise": "normal"
        },
        total_timesteps=20000
    )

    print("📈 Evaluating risk-aware multi-asset agent...")
    env_test = get_env(test, stock_dim, risk_aware=True)
    evaluate_agent(model, env_test, label="MultiAsset_SAC_Risk", save_path="results/multiasset_sac_risk_curve.png")

if __name__ == "__main__":
    main()
