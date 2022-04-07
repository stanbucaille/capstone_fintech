from abc import abstractmethod

from tensortrade import Component
from tensortrade.orders import Trade


class SlippageModel(Component):
    """A model for simulating slippage on an exchange trade."""

    registered_name = "slippage"

    def __init__(self):
        pass

    @abstractmethod
    def adjust_trade(self, trade: Trade, **kwargs) -> Trade:
        """Simulate slippage on a trade ordered on a specific exchange.

        Arguments:
            trade: The trade executed on the exchange.
            **kwargs: Any other arguments necessary for the model.

        Returns:
            A filled `Trade` with the `price` and `size` adjusted for slippage.
        """

        raise NotImplementedError()
