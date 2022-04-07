from tensortrade.orders import Order, OrderSpec, TradeSide, TradeType
from tensortrade.orders.criteria import Stop, Limit


def market_order(step: int,
                 side: 'TradeSide',
                 pair: 'TradingPair',
                 price: float,
                 size: float,
                 portfolio: 'Portfolio'):
    instrument = side.instrument(pair)
    order = Order(step=step,
                  side=side,
                  trade_type=TradeType.MARKET,
                  pair=pair,
                  price=price,
                  quantity=(size * instrument),
                  portfolio=portfolio
                  )

    return order


def limit_order(step: int,
                side: 'TradeSide',
                pair: 'TradingPair',
                price: float,
                size: float,
                portfolio: 'Portfolio',
                start: int = None,
                end: int = None):
    instrument = side.unstrument(pair)
    order = Order(step=step,
                  side=side,
                  trade_type=TradeType.LIMIT,
                  pair=pair,
                  price=price,
                  quantity=(size * instrument),
                  start=start,
                  end=end,
                  portfolio=portfolio
                  )

    return order


def hidden_limit_order(step: int,
                       side: 'TradeSide',
                       pair: 'TradingPair',
                       price: float,
                       size: float,
                       portfolio: 'Portfolio',
                       start: int = None,
                       end: int = None):
    instrument = side.instrument(pair)
    order = Order(step=step,
                  side=side,
                  trade_type=TradeType.MARKET,
                  pair=pair,
                  price=price,
                  quantity=(size * instrument),
                  start=start,
                  end=end,
                  portfolio=portfolio,
                  criteria=Limit(limit_price=price)
                  )

    return order


def risk_managed_order(step: int,
                       side: 'TradeSide',
                       trade_type: 'TradeType',
                       pair: 'TradingPair',
                       price: float,
                       size: float,
                       down_percent: float,
                       up_percent: float,
                       portfolio: 'Portfolio',
                       start: int = None,
                       end: int = None):
    instrument = side.instrument(pair)
    order = Order(step=step,
                  side=side,
                  trade_type=trade_type,
                  pair=pair,
                  price=price,
                  start=start,
                  end=end,
                  quantity=(size * instrument),
                  portfolio=portfolio)
    # ^ :xor. here risk_criteria is a composite criteria
    risk_criteria = Stop("down", down_percent) ^ Stop("up", up_percent)
    # risk_management中的side设置成和开仓方向相反，和止盈止损含义对应
    risk_management = OrderSpec(side=TradeSide.SELL if side == TradeSide.BUY else TradeSide.BUY,
                                trade_type=TradeType.MARKET,
                                pair=pair,
                                criteria=risk_criteria)

    order += risk_management  ##both order and order_spec are inherited from dict

    return order
