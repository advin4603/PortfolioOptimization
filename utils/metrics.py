# utils/metrics.py

import pandas as pd
import numpy as np

def compute_metrics(df_account_value):
    returns = df_account_value['account_value'].pct_change().dropna()
    cum_return = df_account_value['account_value'].iloc[-1] / df_account_value['account_value'].iloc[0] - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252)
    max_drawdown = (df_account_value['account_value'] / df_account_value['account_value'].cummax() - 1).min()

    return {
        "Cumulative Return": cum_return,
        "Sharpe Ratio": sharpe,
        "Volatility": volatility,
        "Max Drawdown": max_drawdown
    }

def summarize_all(results_dict):
    summary = []

    for name, df in results_dict.items():
        metrics = compute_metrics(df)
        metrics["Model"] = name
        summary.append(metrics)

    return pd.DataFrame(summary).sort_values(by="Sharpe Ratio", ascending=False)
