# models/trainer.py

from finrl.agents.stablebaselines3.models import DRLAgent

def train_single_asset_agent(train_df, get_env_fn, symbol, risk_aware=True, algos=["ppo", "a2c"], total_timesteps=30000):
    trained_models = {}
    env_train = get_env_fn(train_df, risk_aware=risk_aware)
    env_train, _ = env_train.get_sb_env()
    agent = DRLAgent(env=env_train)

    for algo in algos:
        print(f"🔁 Training {algo.upper()} for {symbol} (Risk Aware: {risk_aware})")
        model = agent.get_model(model_name=algo)
        model = agent.train_model(model=model, tb_log_name=f"{algo}_{symbol}", total_timesteps=total_timesteps)
        model.save(f"saved_models/{symbol}_{algo}{'_risk' if risk_aware else ''}")
        trained_models[algo] = model

    return trained_models


def train_multi_asset_agent(env_train, algo="ddpg", model_kwargs=None, total_timesteps=20000):
    from finrl.agents.stablebaselines3.models import DRLAgent
    agent = DRLAgent(env=env_train)
    model = agent.get_model(model_name=algo, model_kwargs=model_kwargs or {})
    model = agent.train_model(model=model, tb_log_name=algo, total_timesteps=total_timesteps)
    model.save(f"saved_models/{algo}_multi_asset_risk")
    return model
