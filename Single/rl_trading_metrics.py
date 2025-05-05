import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compute_metrics(df_account_value: pd.DataFrame, name="") -> dict:
    df = df_account_value.copy()
    df['daily_return'] = df['account_value'].pct_change()

    cumulative_return = (df['daily_return'] + 1).cumprod().iloc[-1] - 1
    annualized_return = df['daily_return'].mean() * 252
    volatility = df['daily_return'].std() * np.sqrt(252)

    sharpe_ratio = (
        annualized_return / volatility
        if volatility != 0 else 0
    )

    downside_std = df[df['daily_return'] < 0]['daily_return'].std()
    sortino_ratio = (
        annualized_return / (downside_std * np.sqrt(252))
        if downside_std != 0 else 0
    )

    rolling_max = df['account_value'].cummax()
    drawdown = df['account_value'] / rolling_max - 1.0
    max_drawdown = drawdown.min()

    final_value = df['account_value'].iloc[-1]

    return {
        "strategy": name,
        "final_value": final_value,
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown
    }

def summarize_all(results: dict) -> pd.DataFrame:
    all_metrics = []
    for name, df in results.items():
        metrics = compute_metrics(df, name)
        all_metrics.append(metrics)
    return pd.DataFrame(all_metrics).sort_values(by="sharpe_ratio", ascending=False)

def plot_bar_metrics(summary_df, metric_list=None, title_prefix="Strategy"):
    if metric_list is None:
        metric_list = ["final_value", "cumulative_return", "annualized_return",
                       "sharpe_ratio", "sortino_ratio", "volatility", "max_drawdown"]

    for metric in metric_list:
        plt.figure(figsize=(10, 5))
        sns.barplot(x="strategy", y=metric, data=summary_df)
        plt.title(f"{title_prefix} Comparison - {metric.replace('_', ' ').title()}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(True)
        plt.show()
