import numpy as np

from tensortrade.exchanges.services.slippage import SlippageModel
from tensortrade.orders import Trade, TradeType, TradeSide


class RandomUniformSlippageModel(SlippageModel):
    """A uniform random slippage model."""

    def __init__(self, max_slippage_percent: float = 3.0):
        """
        Arguments:
            max_slippage_percent: The maximum random slippage to be applied to the fill price. Defaults to 3.0 (i.e. 3%).
        """
        self.max_slippage_percent = self.default('max_slippage_percent', max_slippage_percent)

    def adjust_trade(self, trade: Trade) -> Trade:
        price_slippage = np.random.uniform(0, self.max_slippage_percent / 100)

        initial_price = trade.price

        if trade.type == TradeType.MARKET:
            if trade.side == TradeSide.BUY:
                trade.price = max(initial_price * (1 + price_slippage), 1e-3)
            else:
                trade.price = max(initial_price * (1 - price_slippage), 1e-3)
        else:
            if trade.side == TradeSide.BUY:
                trade.price = max(initial_price * (1 + price_slippage), 1e-3)

                if trade.price > initial_price:
                    trade.size *= min(initial_price / trade.price, 1)
            else:
                trade.price = max(initial_price * (1 - price_slippage), 1e-3)

                if trade.price < initial_price:
                    trade.size *= min(trade.price / initial_price, 1)

        return trade
