# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 21:59:30 2020

@author: CycloneMAO
@contact: 877245759@qq.com
"""

import tensortrade.orders.create as create

from typing import Union, List, Tuple
from itertools import product
from gym.spaces import Discrete, Box

from tensortrade.actions import ActionScheme
from tensortrade.orders import TradeSide, TradeType, Order, OrderListener, risk_managed_order
import numpy as np

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

class DiscretePositionScheme(ActionScheme):
    """
    """

    def __init__(self,
                 positionList: Union[List[float], float] = np.arange(0, 1.01, 0.1).tolist(),
                 trade_type: TradeType = TradeType.MARKET,
                 order_listener: OrderListener = None):  #still a virtual class so far, similar call_back functionality to that of broker class
        """
        Arguments:
            positionList: A list of position to hold for a single asset(no negative position so far)
            order_listener (optional): An optional listener for order events executed by this action scheme.
        """
        
        self.positionList = positionList
        self.trade_type = trade_type
        self._order_listener = order_listener

        self._actions = self.positionList

    def computePosition(self, portfolio: 'Portfolio', instrument: 'Instrument', price: float):
        return portfolio.total_balance(instrument).size * price / portfolio.net_worth
      
    def get_order(self, 
                  action: Union[tuple, int], 
                  portfolio: 'Portfolio') -> Union[Order, List[Order]]:
        if action[0] == 0:
            return []
        
        # from action encoding to trading action
        ((exchange, pair), position) = self.actions[action[0]]  
        
        # take current price as the target price and compute the order's target specifications
        target_price = exchange.quote_price(pair) # here use quote_price hence never forward the exchange

        # compute current position of the quoted instrument
        current_pos = self.computePosition(portfolio, pair.quote, target_price)
        dev_pos = position - current_pos
        
        # decide the trading side and the wallet to allocate
        side = TradeSide.BUY if dev_pos > 0 else TradeSide.SELL
        
        allocated_instrument = side.instrument(pair)
        counter_instrument = side.counter_instrument(pair)
        allocated_wallet = portfolio.get_wallet(exchange.id, instrument=allocated_instrument)
        counter_wallet = portfolio.get_wallet(exchange.id, instrument=counter_instrument)
       
        # calculate target trading size
        if side.value == 'buy':
            target_base_size = round_down(abs(dev_pos) * portfolio.net_worth, allocated_instrument.precision)
            #commission is always computed in base
            target_commission_size = round_up(target_base_size * exchange.options.commission / (1 + exchange.options.commission), pair.base.precision)
            target_quote_value = target_base_size - target_commission_size
            target_quote_size = round_down(min(allocated_wallet.balance.size, target_quote_value) / target_price, pair.quote.precision)
        elif side.value == 'sell':
            target_quote_size = round_down(abs(dev_pos) * portfolio.net_worth / target_price, allocated_instrument.precision)
            target_quote_size = min(counter_wallet.balance.size, target_quote_size)
        
        if target_quote_size == 0:
            return []
          
        order = Order(step=exchange.clock.step,
                  side=side,
                  trade_type=self.trade_type,
                  pair=pair,
                  price=target_price,
                  start=exchange.clock.step,
                  end=None,
                  quantity=target_quote_size * (counter_instrument if side == TradeSide.BUY else allocated_instrument),
                  portfolio=portfolio)
        

        if self._order_listener is not None:
            order.attach(self._order_listener)

        return [order]

    def reset(self):
        pass


class ContinuousPositionScheme(ActionScheme):
    """
    """

    def __init__(self,
                 receive_action_type: str = "logit",
                 num_assets: int = 2,
                 trade_type: TradeType = TradeType.MARKET,
                 order_listener: OrderListener = None):  #still a virtual class so far, similar call_back functionality to that of broker class
        """
        Notes:
            1. There's only one base instrument, and each of any other instrument forms a trading pair with it. 
            Hence it's not compatible with exchanges with multiple base instruments(eg, Crypto, Forex)
            
            2. Order queue: submit sell orders before submit buy orders
            
        Arguments:
            receive_action_type: action encoding type
            num_quotes: int, >= 2 (Always assume there exist a base instrument)
            order_listener (optional): An optional listener for order events executed by this action scheme.
        """
        self.receive_action_type = receive_action_type
        self.num_assets = num_assets
        self.trade_type = self.default('trade_type', trade_type)
        self._order_listener = self.default('order_listener', order_listener)
        
        self._actions = [Box(-5, 5, (self.num_assets,))]  # after self.set_pairs, it will be of shape (num_assets)
    
    def computePosition(self, portfolio: 'Portfolio', instrument: 'Instrument', price: float):
        return portfolio.total_balance(instrument).size * price / portfolio.net_worth
      
    def get_order(self, action: int, portfolio: 'Portfolio') -> Order:
        sell_orders, buy_orders = [], []
        if self.receive_action_type == "logit":
            target_position = np.exp(action)/np.sum(np.exp(action))
        for i in range(1, self.num_assets):
            ((exchange, pair), position) = self.actions[i][0], target_position[i]
            
            # take current price as the target price and compute the order's target specifications
            target_price = exchange.quote_price(pair) # here use quote_price hence never forward the exchange
    
            # compute current position of the quoted instrument
            current_pos = self.computePosition(portfolio, pair.quote, target_price)
            dev_pos = position - current_pos
            
            # decide the trading side and the wallet to allocate
            side = TradeSide.BUY if dev_pos > 0 else TradeSide.SELL
            
            allocated_instrument = side.instrument(pair)
            counter_instrument = side.counter_instrument(pair)
            allocated_wallet = portfolio.get_wallet(exchange.id, instrument=allocated_instrument)
            counter_wallet = portfolio.get_wallet(exchange.id, instrument=counter_instrument)
           
            # calculate target trading size
            if side.value == 'buy':
                target_base_size = round_down(abs(dev_pos) * portfolio.net_worth, allocated_instrument.precision)
                #commission is always computed in base
                target_commission_size = round_up(target_base_size * exchange.options.commission / (1 + exchange.options.commission), pair.base.precision)
                target_quote_value = target_base_size - target_commission_size
                target_quote_size = round_down(min(allocated_wallet.balance.size, target_quote_value) / target_price, pair.quote.precision)
            elif side.value == 'sell':
                target_quote_size = round_down(abs(dev_pos) * portfolio.net_worth / target_price, allocated_instrument.precision)
                target_quote_size = min(counter_wallet.balance.size, target_quote_size)
            
            if target_quote_size == 0:
                continue
              
            order = Order(step=exchange.clock.step,
                      side=side,
                      trade_type=self.trade_type,
                      pair=pair,
                      price=target_price,
                      start=exchange.clock.step,
                      end=None,
                      quantity=target_quote_size * (counter_instrument if side == TradeSide.BUY else allocated_instrument),
                      portfolio=portfolio)
            
    
            if self._order_listener is not None:
                order.attach(self._order_listener)
            
            if side.value == "sell":
                sell_orders.append(order)
            elif side.value == "buy":
                buy_orders.append(order)
                
        return sell_orders + buy_orders

    def reset(self):
        pass