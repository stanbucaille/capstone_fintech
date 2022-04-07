# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 18:42:54 2022

@author: Jianmin Mao
@email: 877245759@qq.com
@tel: (86) 18194038783
"""
import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context # Only used if pandas gives an SSLError
def fetch_cryptodatadownload(exchange_name, symbol, base, timeframe):
    """
    

    Parameters
    ----------
    exchange_name : TYPE
        DESCRIPTION.
    symbol : TYPE
        DESCRIPTION.
    base : TYPE
        DESCRIPTION.
    timeframe : TYPE   "minute" "1h" "d"
        DESCRIPTION.

    """
    url = "https://www.cryptodatadownload.com/cdd/"
    filename = "{}_{}{}_{}.csv".format(exchange_name, symbol, base, timeframe)
    df = pd.read_csv(url + filename, skiprows=1)
    df = df[::-1]
    df = df.drop(["unix", "symbol", "Volume %s"%symbol], axis=1)
    df = df.rename({"Volume USDT": "volume"}, axis=1)
    df = df.set_index("date")
    df.columns = [symbol + ":" + name.lower() for name in df.columns]
    return df