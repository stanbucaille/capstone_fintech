import numpy as np

from tensortrade.stochastic.processes.brownian_motion import brownian_motion_log_returns
from tensortrade.stochastic.utils import ModelParameters, generate


def cox_ingersoll_ross_levels(params):
    """
    Constructs the rate levels of a mean-reverting Cox-Ingersoll-Ross process.
    Used to model interest rates as well as stochastic volatility in the Heston
    model. We pass a correlated Brownian motion process into the method from
    which the interest rate levels are constructed because the returns between
    the underlying and the stochastic volatility should be correlated. The other
     correlated process is used in the Heston model.

    Arguments:
        params : ModelParameters
            The parameters for the stochastic model.
    Returns:
        The interest rate levels for the CIR process.
    """
    brownian_motion = brownian_motion_log_returns(params)
    # Setup the parameters for interest rates
    a, mu, zero = params.cir_a, params.cir_mu, params.all_r0
    # Assumes output is in levels
    levels = [zero]
    for i in range(1, params.all_time):
        drift = a * (mu - levels[i - 1]) * params.all_delta
        # The main difference between this and the Ornstein Uhlenbeck model is that we multiply the 'random'
        # component by the square-root of the previous level i.e. the process has level dependent interest rates.
        randomness = np.sqrt(levels[i - 1]) * brownian_motion[i - 1]
        levels.append(levels[i - 1] + drift + randomness)
    return np.array(levels)


def cox(base_price: int = 1,
        base_volume: int = 1,
        start_date: str = '2010-01-01',
        start_date_format: str = '%Y-%m-%d',
        times_to_generate: int = 1000,
        time_frame: str = '1h',
        params: ModelParameters = None):

    data_frame = generate(
        price_fn=cox_ingersoll_ross_levels,
        base_price=base_price,
        base_volume=base_volume,
        start_date=start_date,
        start_date_format=start_date_format,
        times_to_generate=times_to_generate,
        time_frame=time_frame,
        params=params
    )

    return data_frame
