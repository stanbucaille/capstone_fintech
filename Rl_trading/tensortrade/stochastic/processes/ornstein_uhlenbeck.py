import numpy as np
import pandas as pd

from stochastic.noise import GaussianNoise

from tensortrade.stochastic.processes.brownian_motion import brownian_motion_log_returns
from tensortrade.stochastic.utils.helpers import get_delta, scale_times_to_generate
from tensortrade.stochastic.utils.parameters import ModelParameters, default


def ornstein_uhlenbeck_levels(params):
    """
    Constructs the rate levels of a mean-reverting ornstein uhlenbeck process.

    Arguments:
        params : ModelParameters
            The parameters for the stochastic model.

    Returns:
        The interest rate levels for the Ornstein Uhlenbeck process
    """
    ou_levels = [params.all_r0]
    brownian_motion_returns = brownian_motion_log_returns(params)
    for i in range(1, params.all_time):
        drift = params.ou_a * (params.ou_mu - ou_levels[i - 1]) * params.all_delta
        randomness = brownian_motion_returns[i - 1]
        ou_levels.append(ou_levels[i - 1] + drift + randomness)
    return np.array(ou_levels)


def ornstein(base_price: int = 1,
             base_volume: int = 1,
             start_date: str = '2010-01-01',
             start_date_format: str = '%Y-%m-%d',
             times_to_generate: int = 1000,
             time_frame: str = '1h',
             params: ModelParameters = None):

    delta = get_delta(time_frame)
    times_to_generate = scale_times_to_generate(times_to_generate, time_frame)

    params = params or default(base_price, times_to_generate, delta)

    prices = ornstein_uhlenbeck_levels(params)

    volume_gen = GaussianNoise(t=times_to_generate)
    volumes = volume_gen.sample(times_to_generate) + base_volume

    start_date = pd.to_datetime(start_date, format=start_date_format)
    price_frame = pd.DataFrame([], columns=['date', 'price'], dtype=float)
    volume_frame = pd.DataFrame([], columns=['date', 'volume'], dtype=float)

    price_frame['date'] = pd.date_range(start=start_date, periods=times_to_generate, freq="1min")
    price_frame['price'] = abs(prices)

    volume_frame['date'] = price_frame['date'].copy()
    volume_frame['volume'] = abs(volumes)

    price_frame.set_index('date')
    price_frame.index = pd.to_datetime(price_frame.index, unit='m', origin=start_date)

    volume_frame.set_index('date')
    volume_frame.index = pd.to_datetime(volume_frame.index, unit='m', origin=start_date)

    data_frame = price_frame['price'].resample(time_frame).ohlc()
    data_frame['volume'] = volume_frame['volume'].resample(time_frame).sum()

    return data_frame
