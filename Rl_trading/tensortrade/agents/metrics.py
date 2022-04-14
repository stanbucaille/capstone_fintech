import numpy as np
import pandas as pd


def annual_return(series):
    return series.mean() * 252


def annual_volatility(series):
    return series.std() * 252 ** 0.5


def annual_sharpe(series):
    return np.sqrt(252) * series.mean() / series.std()


def win_rate_per_day(series):
    return (series > 0).sum() / len(series)


def max_dropdown(series):
    _acc = series.cumsum().tolist()
    _max = -np.inf
    max_dd = 0

    for _val in _acc:
        _max = max(_max, _val)
        max_dd = max(max_dd, _max - _val)
    return max_dd


def compute_metrics(ret_df, metrics=["annual_return", "annual_sharpe", "max_dropdown", "win_rate_per_day"]):
    result = []
    for metric in metrics:
        result.append(ret_df.apply(lambda x: eval(metric)(x)))

    result = pd.concat(result, axis=1)
    result.columns = metrics
    result.index = ret_df.columns
    return result

