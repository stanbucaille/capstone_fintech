from tensortrade.orders.criteria import Criteria
from tensortrade.orders import TradeSide


class Limit(Criteria):
    """An order criteria that allows execution when the quote price for a
    trading pair is at or below a specific price, hidden from the public order book."""

    def __init__(self, limit_price: float):
        self.limit_price = limit_price

    def check(self, order: 'Order', exchange: 'Exchange') -> bool:
        price = exchange.quote_price(order.pair)

        buy_satisfied = (order.side == TradeSide.BUY and price <= self.limit_price)
        sell_satisfied = (order.side == TradeSide.SELL and price >= self.limit_price)

        return buy_satisfied or sell_satisfied

    def __str__(self):
        return '<Limit: price={0}>'.format(self.limit_price)
