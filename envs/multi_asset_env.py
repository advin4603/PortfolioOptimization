import numpy as np
import pandas as pd
import gym
from gym import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from gym.utils import seeding
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class MultiAssetStockTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, stock_dim, hmax, initial_amount, transaction_cost_pct,
                 reward_scaling, state_space, action_space, tech_indicator_list,
                 turbulence_threshold=None, lookback=252, day=0):
        self.day = day
        self.lookback = lookback
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list

        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.state_space + len(self.tech_indicator_list), self.state_space))

        self.data = self.df.loc[self.day, :]
        self.covs = self.data['cov_list'].values[0]
        self.state = np.append(np.array(self.covs),
                               [self.data[tech].values.tolist() for tech in self.tech_indicator_list], axis=0)

        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            df = pd.DataFrame(self.portfolio_return_memory, columns=['daily_return'])

            plt.plot(df.daily_return.cumsum(), 'r')
            plt.savefig('results/cumulative_reward.png')
            plt.close()

            plt.plot(df.daily_return, 'r')
            plt.savefig('results/rewards.png')
            plt.close()

            print("=================================")
            print(f"begin_total_asset: {self.asset_memory[0]}")
            print(f"end_total_asset: {self.portfolio_value}")
            if df['daily_return'].std() != 0:
                sharpe = (252 ** 0.5) * df['daily_return'].mean() / df['daily_return'].std()
                print("Sharpe Ratio:", sharpe)
            print("=================================")

            return self.state, self.reward, self.terminal, {}

        else:
            weights = self.softmax_normalization(actions)
            self.actions_memory.append(weights)
            last_day_memory = self.data

            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.covs = self.data['cov_list'].values[0]
            self.state = np.append(np.array(self.covs),
                               [self.data[tech].values.tolist() for tech in self.tech_indicator_list], axis=0)


            portfolio_return = sum(((self.data.close.values / last_day_memory.close.values) - 1) * weights)
            new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
            self.portfolio_value = new_portfolio_value

            self.portfolio_return_memory.append(portfolio_return)
            self.asset_memory.append(new_portfolio_value)
            self.date_memory.append(self.data.date.unique()[0])

            # ✅ Risk-aware reward based on 20-day Sharpe ratio
            if len(self.portfolio_return_memory) > 20:
                returns = np.array(self.portfolio_return_memory[-20:])
                std = np.std(returns)
                mean = np.mean(returns)
                self.reward = (mean / (std + 1e-8)) * 100 if std > 0 else portfolio_return
            else:
                self.reward = portfolio_return

            return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.covs = self.data['cov_list'].values[0]
        self.state = np.append(np.array(self.covs),
                               [self.data[tech].values.tolist() for tech in self.tech_indicator_list], axis=0)

        self.portfolio_value = self.initial_amount
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        return self.state

    def render(self, mode='human'):
        return self.state

    def softmax_normalization(self, actions):
        exp_actions = np.exp(actions)
        return exp_actions / np.sum(exp_actions)

    def save_asset_memory(self):
        return pd.DataFrame({'date': self.date_memory, 'daily_return': self.portfolio_return_memory})

    def save_action_memory(self):
        df_date = pd.DataFrame(self.date_memory, columns=['date'])
        df_actions = pd.DataFrame(self.actions_memory)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
