from datetime import datetime
from itertools import product
from typing import Union, List, Dict

from tensortrade.base.core import TimeIndexed

from .order import Order, OrderStatus
from .order_listener import OrderListener


class Broker(OrderListener, TimeIndexed):
    """A broker for handling the execution of orders on multiple exchanges.
    Orders are kept in a virtual order book until they are ready to be executed.
    """

    def __init__(self, exchanges: Union[List['Exchange'], 'Exchange'], on_execute_verbose = True):
        self.exchanges = exchanges

        self._unexecuted = []
        self._executed = {}
        self._trades = {}
        self._on_execute_verbose = on_execute_verbose

    @property
    def exchanges(self) -> List['Exchange']:
        """The list of exchanges the broker will execute orders on."""
        return self._exchanges

    @exchanges.setter
    def exchanges(self, exchanges: Union[List['Exchange'], 'Exchange']):
        self._exchanges = exchanges if isinstance(exchanges, list) else [exchanges]

    @property
    def unexecuted(self) -> List[Order]:
        """The list of orders the broker is waiting to execute, when their criteria is satisfied."""
        return self._unexecuted

    @property
    def executed(self) -> Dict[str, Order]:
        """The dictionary of orders the broker has executed since resetting, organized by order id"""
        return self._executed

    @property
    def trades(self) -> Dict[str, 'Trade']:
        """The dictionary of trades the broker has executed since resetting, organized by order id."""
        return self._trades

    def submit(self, order: Order):
        self._unexecuted += [order]

    def cancel(self, order: Order):
        if order.status == OrderStatus.CANCELLED:
            raise Warning(
                'Cannot cancel order {} - order has already been cancelled.'.format(order.id))

        if order.status != OrderStatus.PENDING:
            raise Warning(
                'Cannot cancel order {} - order has already been executed.'.format(order.id))

        self._unexecuted.remove(order)

        order.cancel()

    def update(self):
        
        # execute orders if possible
        for order, exchange in product(self._unexecuted, self._exchanges):
            #check:
            #1.Does the order satisfy its own criteria(order, exchange)
            #2.Is the order expired
            is_executable = order.is_executable_on(exchange) and self.clock.step == order.start

            if order in self._unexecuted and is_executable:
                self._unexecuted.remove(order)
                self._executed[order.id] = order

                order.attach(self) #here this order attachs the broker in env so that that broker can use on_fill to create this order's own exit order
                order.execute(exchange)
        
        # update orders it manages
        for order in self._unexecuted + list(self._executed.values()):  #will go through all the orders the broker has ever met, which is time-consuming -->need optimization
            order_expired = (self.clock.step >= order.end) if order.end else True  #by default, order.end=None, hence expired immediately after execution(no matter executbale or not)

            order_active = order.status not in [OrderStatus.FILLED, OrderStatus.CANCELLED]

            if order_active and order_expired:
                order.cancel()  #turn its status to cancelled and release the wallet

    def on_fill(self, order: Order, exchange: 'Exchange', trade: 'Trade'):

        self._trades[trade.order_id] = self._trades.get(trade.order_id, [])
        self._trades[trade.order_id] += [trade]

        if order.is_complete():
            next_order = order.complete(exchange)

            if next_order:
                self.submit(next_order)  # exiting-position orders

    def on_execute(self, order, exchange):  #right before updating the account
        if self._on_execute_verbose:
            execution_message = {
              "order_side": order.side,
              "order_quantity": order.quantity,
              "order_price": order.price,
              "order_create_at": order.created_at
              }
            print(execution_message)
            
    def reset(self):
        self._unexecuted = []
        self._executed = {}
        self._trades = {}
