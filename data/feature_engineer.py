# data/feature_engineer.py

import pandas as pd
from finrl.meta.preprocessor.preprocessors import FeatureEngineer

def add_cov_and_returns(df, lookback=252):
    df = df.sort_values(['date','tic'],ignore_index=True)
    df.index = df.date.factorize()[0]  # numeric index

    cov_list = []
    return_list = []

    for i in range(lookback, len(df.index.unique())):
        data_lookback = df.loc[i - lookback:i]
        price_lookback = data_lookback.pivot_table(index='date', columns='tic', values='close')
        return_lookback = price_lookback.pct_change().dropna()
        return_list.append(return_lookback)
        covs = return_lookback.cov().values
        cov_list.append(covs)

    df_cov = pd.DataFrame({
        'date': df.date.unique()[lookback:],
        'cov_list': cov_list,
        'return_list': return_list
    })

    df = df.merge(df_cov, on='date')
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    return df

# Expose FeatureEngineer class from FinRL
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
