import numpy as np

from tensortrade.stochastic.utils import ModelParameters, convert_to_prices


def brownian_motion_log_returns(params: ModelParameters):
    """
    Constructs a Wiener process (Brownian Motion).

    References:
        - http://en.wikipedia.org/wiki/Wiener_process

    Arguments:
        params : ModelParameters
            The parameters for the stochastic model.

    Returns:
        brownian motion log returns
    """
    sqrt_delta_sigma = np.sqrt(params.all_delta) * params.all_sigma
    return np.random.normal(loc=0, scale=sqrt_delta_sigma, size=params.all_time)


def brownian_motion_levels(params: ModelParameters):
    """
    Constructs a price sequence whose returns evolve according to brownian
    motion.

    Arguments:
        params : ModelParameters
            The parameters for the stochastic model.

    Returns:
        A price sequence which follows brownian motion
    """
    return convert_to_prices(params, brownian_motion_log_returns(params))
