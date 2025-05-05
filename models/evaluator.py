# models/evaluator.py

from finrl.agents.stablebaselines3.models import DRLAgent
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_agent(model, env, label=None, save_path=None):
    df_account_value, _ = DRLAgent.DRL_prediction(model=model, environment=env)
    df_account_value.set_index('date', inplace=True)
    df_account_value.index = pd.to_datetime(df_account_value.index)

    plt.figure(figsize=(12, 6))
    plt.plot(df_account_value['account_value'], label=label or "Agent")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.title(f"Performance: {label}")
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

    return df_account_value
