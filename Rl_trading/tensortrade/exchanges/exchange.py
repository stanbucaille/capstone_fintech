from typing import Callable, Union

from tensortrade.base import Component, TimedIdentifiable
from tensortrade.instruments import TradingPair
from tensortrade.data import Module
from tensortrade.data import Forward


class ExchangeOptions:

    def __init__(self,
                 commission: float = 0.002,
                 min_trade_size: float = 0,
                 max_trade_size: float = float("inf"),
                 min_trade_price: float = 1e-8,
                 max_trade_price: float = 1e8,
                 is_live: bool = False):
        self.commission = commission
        self.min_trade_size = min_trade_size
        self.max_trade_size = max_trade_size  # in terms of quote size; may need to specified as a dictionary in order to distinguish multiple tradable pairs
        self.min_trade_price = min_trade_price
        self.max_trade_price = max_trade_price
        self.is_live = is_live


class Exchange(Module, Component, TimedIdentifiable):
    """An abstract exchange for use within a trading environment."""

    registered_name = "exchanges"

    def __init__(self,
                 name: str,
                 service: Union[Callable, str],
                 options: ExchangeOptions = None):
        super().__init__(name)

        self._service = service
        self._options = options if options else ExchangeOptions()
        self._prices = None
           
        #不知道为什么__init__用不了build
        #self.build()  ##不build不会产生任何_prices, order need this info to do transaction

    @property
    def options(self):
        return self._options

    def build(self):   ###need to build if using other base_instrument, eg:CNY
        if self.built:
            return
        self._prices = {}

        for node in self.inputs:
            pair = "".join([c if (c.isalnum() or c == '.' or c == '_') else "/" for c in node.name])
            self._prices[pair] = Forward(node)
        self.built = True
        
    def quote_price(self, trading_pair: 'TradingPair') -> float:
        """The quote price of a trading pair on the exchange, denoted in the base instrument.

        Arguments:
            trading_pair: The `TradingPair` to get the quote price for.

        Returns:
            The quote price of the specified trading pair, denoted in the base instrument.
        """
        #print(self.inputs[0]._cursor)
        return self._prices[str(trading_pair)].forward()  #here won't update the static Forward node

    def is_pair_tradable(self, trading_pair: 'TradingPair') -> bool:
        """Whether or not the specified trading pair is tradable on this exchange.

        Args:
            trading_pair: The `TradingPair` to test the tradability of.

        Returns:
            A bool designating whether or not the pair is tradable.
        """
        return str(trading_pair) in self._prices.keys()

    def execute_order(self, order: 'Order', portfolio: 'Portfolio'):
        """Execute an order on the exchange.

        Arguments:
            order: The order to execute.
            portfolio: The portfolio to use.
        """
        trade = self._service(
            order=order,
            base_wallet=portfolio.get_wallet(self.id, order.pair.base),
            quote_wallet=portfolio.get_wallet(self.id, order.pair.quote),
            current_price=self.quote_price(order.pair),  # achieve current price for execution
            options=self.options,
            exchange_id=self.id,
            clock=self.clock
        )

        if trade:
            order.fill(self, trade)

    #由于trading_env会自动生成internal feed,会将portfolio中的exchange里面的Stream自动纳入到数据流进行更新（即.next()方法），所以下面这个单独更新exchange的函数暂时用不到
    def _update(self):
        for i in range(len(self.inputs)):
            self.inputs[i].run()
                
    def has_next(self):
        for i in range(len(self.inputs)):
            if hasattr(self.inputs[i], "_array"): #i.e., whether the node is a Stream
                return self.inputs[i].has_next()

    def reset(self):
        for i in range(len(self.inputs)):
            self.inputs[i].reset() #只重置cursor，value还是之前的
        
            
          
