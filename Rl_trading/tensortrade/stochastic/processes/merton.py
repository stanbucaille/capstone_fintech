import pandas as pd

from stochastic.noise import GaussianNoise

from tensortrade.stochastic.processes.heston import geometric_brownian_motion_jump_diffusion_levels
from tensortrade.stochastic.utils.helpers import get_delta, scale_times_to_generate
from tensortrade.stochastic.utils.parameters import ModelParameters, default


def merton(base_price: int = 1,
           base_volume: int = 1,
           start_date: str = '2010-01-01',
           start_date_format: str = '%Y-%m-%d',
           times_to_generate: int = 1000,
           time_frame: str = '1h',
           params: ModelParameters = None):

    delta = get_delta(time_frame)
    times_to_generate = scale_times_to_generate(times_to_generate, time_frame)

    params = params or default(base_price, times_to_generate, delta)

    prices = geometric_brownian_motion_jump_diffusion_levels(params)

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
