import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_jsonl(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)

def calculate_metrics(df):
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df.set_index('date', inplace=True)

    # Strategy Returns and Sharpe Ratio
    df['returns'] = df['net_worth'].pct_change()
    sharpe_strategy = df['returns'].mean() / df['returns'].std() * np.sqrt(252)

    # Stock Returns and Sharpe Ratio
    stock_returns = df['close'].pct_change()
    sharpe_stock = stock_returns.mean() / stock_returns.std() * np.sqrt(252)

    # Drawdown
    cumulative = df['net_worth'].cummax()
    drawdown = df['net_worth'] / cumulative - 1
    max_drawdown = drawdown.min()

    # Trade stats
    buy_count = df[df['action'] > 0].shape[0]
    sell_count = df[df['action'] < 0].shape[0]

    # Performance summary
    metrics = {
        "Final Net Worth": df['net_worth'].iloc[-1],
        "Total Return (%)": ((df['net_worth'].iloc[-1] - df['net_worth'].iloc[0]) / df['net_worth'].iloc[0]) * 100,
        "Sharpe Ratio (Strategy)": sharpe_strategy,
        "Sharpe Ratio (Stock)": sharpe_stock,
        "Max Drawdown (%)": max_drawdown * 100,
        "Buy Count": buy_count,
        "Sell Count": sell_count,
    }

    return df, metrics

def plot(df):
    fig, ax = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

    # Net Worth Plot
    ax[0].plot(df.index, df['net_worth'], label='Net Worth', color='blue')
    ax[0].set_ylabel("Net Worth")
    ax[0].legend()
    ax[0].grid(True)

    # Close Price Plot
    ax[1].plot(df.index, df['close'], label='Close Price', color='orange')
    ax[1].set_ylabel("Close Price")
    ax[1].legend()
    ax[1].grid(True)

    # Action Plot
    colors = df['action'].apply(lambda x: 'green' if x > 0 else 'red' if x < 0 else 'gray')
    ax[2].bar(df.index, df['action'], color=colors)
    ax[2].set_ylabel("Action (Shares)")
    ax[2].set_xlabel("Date")
    ax[2].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_jsonl("trading_run_no_chat_history.jsonl")
    df, metrics = calculate_metrics(df)

    print("=== Performance Summary ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")

    plot(df)
