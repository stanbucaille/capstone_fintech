import tensortrade.orders.create as create

from typing import Union, List, Tuple
from itertools import product
from gym.spaces import Discrete

from tensortrade.actions import ActionScheme
from tensortrade.orders import TradeSide, TradeType, Order, OrderListener, risk_managed_order


class ManagedRiskOrders(ActionScheme):
    """A discrete action scheme that determines actions based on managing risk,
       through setting a follow-up stop loss and take profit on every order.
    """

    def __init__(self,
                 stop_loss_percentages: Union[List[float], float] = [0.02, 0.04, 0.06],
                 take_profit_percentages: Union[List[float], float] = [0.01, 0.02, 0.03],
                 trade_sizes: Union[List[float], int] = 10,
                 trade_type: TradeType = TradeType.MARKET,
                 duration: int = None,
                 order_listener: OrderListener = None):
        """
        Arguments:
            pairs: A list of trading pairs to select from when submitting an order.
            (e.g. TradingPair(BTC, USD), TradingPair(ETH, BTC), etc.)
            stop_loss_percentages: A list of possible stop loss percentages for each order.
            take_profit_percentages: A list of possible take profit percentages for each order.
            trade_sizes: A list of trade sizes to select from when submitting an order.
            (e.g. '[1, 1/3]' = 100% or 33% of balance is tradable. '4' = 25%, 50%, 75%, or 100% of balance is tradable.)
            order_listener (optional): An optional listener for order events executed by this action scheme.
        """
        self.stop_loss_percentages = self.default('stop_loss_percentages', stop_loss_percentages)
        self.take_profit_percentages = self.default(
            'take_profit_percentages', take_profit_percentages)
        self.trade_sizes = self.default('trade_sizes', trade_sizes)
        self.trade_type = self.default('trade_type', trade_type)
        self.duration = self.default('duration', duration)
        self._order_listener = self.default('order_listener', order_listener)

        generator = product(self.stop_loss_percentages,
                            self.take_profit_percentages,
                            self.trade_sizes,
                            [TradeSide.BUY, TradeSide.SELL])
        self.actions = list(generator)

    @property
    def action_space(self) -> Discrete:
        """The discrete action space produced by the action scheme."""
        return Discrete(len(self.actions))

    @property
    def stop_loss_percentages(self) -> List[float]:
        """A list of order percentage losses to select a stop loss from when submitting an order.
        (e.g. 0.01 = sell if price drops 1%, 0.15 = 15%, etc.)
        """
        return self._stop_loss_percentages

    @stop_loss_percentages.setter
    def stop_loss_percentages(self, stop_loss_percentages: Union[List[float], float]):
        self._stop_loss_percentages = stop_loss_percentages if isinstance(
            stop_loss_percentages, list) else [stop_loss_percentages]

    @property
    def take_profit_percentages(self) -> List[float]:
        """A list of order percentage gains to select a take profit from when submitting an order.
        (e.g. 0.01 = sell if price rises 1%, 0.15 = 15%, etc.)
        """
        return self._take_profit_percentages

    @take_profit_percentages.setter
    def take_profit_percentages(self, take_profit_percentages: Union[List[float], float]):
        self._take_profit_percentages = take_profit_percentages if isinstance(
            take_profit_percentages, list) else [take_profit_percentages]

    @property
    def trade_sizes(self) -> List[float]:
        """A list of trade sizes to select from when submitting an order.
        (e.g. '[1, 1/3]' = 100% or 33% of balance is tradable. '4' = 25%, 50%, 75%, or 100% of balance is tradable.)
        """
        return self._trade_sizes

    @trade_sizes.setter
    def trade_sizes(self, trade_sizes: Union[List[float], int]):
        self._trade_sizes = trade_sizes if isinstance(trade_sizes, list) else [
            (x + 1) / trade_sizes for x in range(trade_sizes)]

    def get_order(self, action: int, portfolio: 'Portfolio') -> Order:
        if action == 0:
            return None

        ((exchange, pair), (stop_loss, take_profit, size, side)) = self.actions[action]
        ##why this is 6D?
        ##here use quote_price hence never forward the exchange
        price = exchange.quote_price(pair)

        wallet_instrument = side.instrument(pair)
        wallet = portfolio.get_wallet(exchange.id, instrument=wallet_instrument)

        size = (wallet.balance.size * size)  #won't use the locked quantity
        size = min(wallet.balance.size, size)

        if side.value == 'buy' and size < 10 ** -pair.base.precision:     
            return None
        elif side.value == 'sell' and size < 10 ** -pair.quote.precision:
            return None
          
        params = {
            'step': exchange.clock.step,
            'side': side,
            'pair': pair,
            'price': price,
            'size': size,
            'down_percent': stop_loss,
            'up_percent': take_profit,
            'portfolio': portfolio,
            'trade_type': self.trade_type,
            'end': exchange.clock.step + self.duration if self.duration else None  ##need to check
        }

        order = risk_managed_order(**params)

        if self._order_listener is not None:
            order.attach(self._order_listener)

        return order

    def reset(self):
        pass
