from typing import Callable

from tensortrade.base import Identifiable
from tensortrade.orders import Order, TradeSide, TradeType


class OrderSpec(Identifiable):

    def __init__(self,
                 side: TradeSide,
                 trade_type: TradeType,
                 pair: 'TradingPair',
                 criteria: Callable[['Order', 'Exchange'], bool] = None):
        self.side = side
        self.type = trade_type
        self.pair = pair
        self.criteria = criteria

    def create_order(self, order: 'Order', exchange: 'Exchange') -> 'Order':
        wallet_instrument = self.side.instrument(self.pair)

        wallet = order.portfolio.get_wallet(exchange.id, instrument=wallet_instrument)
        quantity = wallet.locked.get(order.path_id, 0)
        if quantity.size == 0:
            print("shit!!")

        return Order(step=exchange.clock.step,
                     side=self.side,
                     trade_type=self.type,
                     pair=self.pair,
                     quantity=quantity,
                     portfolio=order.portfolio,
                     price=order.price,
                     criteria=self.criteria,
                     start=order.start,
                     end=order.end,
                     path_id=order.path_id)

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "pair": self.pair,
            "criteria": self.criteria
        }

    def __str__(self):
        data = ['{}={}'.format(k, v) for k, v in self.to_dict().items()]
        return '<{}: {}>'.format(self.__class__.__name__, ', '.join(data))

    def __repr__(self):
        return str(self)
