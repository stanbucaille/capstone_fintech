
from tensortrade.base import Clock
from tensortrade.base.exceptions import InsufficientFunds
from tensortrade.wallets import Wallet
from tensortrade.instruments import Quantity
from tensortrade.exchanges import ExchangeOptions
from tensortrade.orders import Order, Trade, TradeType, TradeSide

from math import floor, ceil

def round_down(num, precision):
    temp = pow(10, precision)
    return floor(num*temp)/temp

def round_up(num, precision):
    temp = pow(10, precision)
    return ceil(num*temp)/temp

def contain_price(price: float, options: 'ExchangeOptions') -> float:
    return max(min(price, options.max_trade_price), options.min_trade_price)


def contain_size(size: float, options: 'ExchangeOptions') -> float:
    return max(min(size, options.max_trade_size), options.min_trade_size)


def execute_buy_order(order: 'Order',
                      base_wallet: 'Wallet',
                      quote_wallet: 'Wallet',
                      current_price: float,
                      options: 'ExchangeOptions',
                      exchange_id: str,
                      clock: 'Clock') -> 'Trade':
    price = contain_price(current_price, options)
    
    # if the order is a limit order and the order price is not satisfied
    if order.is_limit_order and price > order.price:
        return None
    
    # compute maximum allocatable base size
    if order.is_market_order:
        path_id = None  # no locked quantity in wallet for a market order
        max_allocated_base = base_wallet.balance.size 
    elif order.is_limit_order:
        path_id = order.path_id
        max_allocated_base = base_wallet.locked[path_id].size
        
        
    # compute maximum ordering capacity and determine the actual quote size to be executed
    max_commission_size = round_up(max_allocated_base * options.commission / (1 + options.commission), order.pair.base.precision)
    max_target_quote_value = max_allocated_base - max_commission_size
    max_target_quote_size = round_down(max_target_quote_value / price, order.pair.quote.precision)
    actual_quote_size = round_down(min(max_target_quote_size, order.size), order.pair.quote.precision)
    
    # if the actual quote size doesn't satisfy the trading requirement
    if actual_quote_size == 0 or actual_quote_size != contain_size(actual_quote_size, options):
        return None
    
    # update base wallet
    actual_quote_value = actual_quote_size * price
    actual_quote_value_quantity = Quantity(order.pair.base, actual_quote_value, path_id)
    actual_commission_size = round_up(actual_quote_value * options.commission, order.pair.base.precision)
    actual_commission = Quantity(order.pair.base, actual_commission_size, path_id)
   
    actual_cost = actual_commission + actual_quote_value_quantity
    base_wallet -= actual_cost
    
    #update quote wallet
    actual_order_quantity = Quantity(order.pair.quote, actual_quote_size, path_id)
    quote_wallet += actual_order_quantity

    trade = Trade(order_id=order.path_id,
                  exchange_id=exchange_id,
                  step=clock.step,
                  pair=order.pair,
                  side=TradeSide.BUY,
                  trade_type=order.type,
                  quantity=actual_order_quantity,
                  price=price,
                  commission=actual_commission)

    return trade


def execute_sell_order(order: 'Order',
                       base_wallet: 'Wallet',
                       quote_wallet: 'Wallet',
                       current_price: float,
                       options: 'ExchangeOptions',
                       exchange_id: str,
                       clock: 'Clock') -> 'Trade':
    price = contain_price(current_price, options)
    path_id = order.path_id
    # if the order is a limit order and the order price is not satisfied
    if order.is_limit_order and price < order.price:
        return None
    
    # compute maximum allocatable quote size and actual quote size to be executed
    max_allocated_quote = quote_wallet.balance.size 
    actual_quote_size = round_down(min(max_allocated_quote, order.size), order.pair.quote.precision)
    
    # if the actual quote size doesn't satisfy the trading requirement
    if actual_quote_size == 0 or actual_quote_size != contain_size(actual_quote_size, options):
        return None
    
    #update quote wallet
    actual_order_quantity = Quantity(order.pair.quote, actual_quote_size, path_id)
    quote_wallet -= actual_order_quantity
    
    # update base wallet
    actual_quote_value = actual_quote_size * price
    actual_base_quantity = Quantity(order.pair.base, actual_quote_value, path_id)
    actual_commission_size = round_up(actual_quote_value * options.commission, order.pair.base.precision)
    actual_commission = Quantity(order.pair.base, actual_commission_size, path_id)
   
    actual_base_chg = actual_base_quantity - actual_commission
    base_wallet += actual_base_chg
    

    trade = Trade(order_id=order.path_id,
                  exchange_id=exchange_id,
                  step=clock.step,
                  pair=order.pair,
                  side=TradeSide.SELL,
                  trade_type=order.type,
                  quantity=actual_order_quantity,
                  price=price,
                  commission=actual_commission)

    return trade


def execute_order(order: 'Order',
                  base_wallet: 'Wallet',
                  quote_wallet: 'Wallet',
                  current_price: float,
                  options: 'Options',
                  exchange_id: str,
                  clock: 'Clock') -> 'Trade':

    if order.is_buy:
        trade = execute_buy_order(
            order=order,
            base_wallet=base_wallet,
            quote_wallet=quote_wallet,
            current_price=current_price,
            options=options,
            exchange_id=exchange_id,
            clock=clock
        )
    elif order.is_sell:
        trade = execute_sell_order(
            order=order,
            base_wallet=base_wallet,
            quote_wallet=quote_wallet,
            current_price=current_price,
            options=options,
            exchange_id=exchange_id,
            clock=clock
        )
    else:
        trade = None

    return trade
