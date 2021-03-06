from typing import Dict, Tuple

from tensortrade.base import Identifiable
from tensortrade.base.exceptions import InsufficientFunds
from tensortrade.instruments import Quantity


class Wallet(Identifiable):
    """A wallet stores the balance of a specific instrument on a specific exchange."""

    def __init__(self, exchange: 'Exchange', quantity: 'Quantity'):
        self._exchange = exchange
        #from Adam King:
        self._initial_size = quantity.size
        self._instrument = quantity.instrument
        self._balance = quantity
        self._locked = {}

    @classmethod
    def from_tuple(cls, wallet_tuple: Tuple['Exchange', 'Instrument', float]):
        exchange, instrument, balance = wallet_tuple
        return cls(exchange, Quantity(instrument, balance))

    @property
    def exchange(self) -> 'Exchange':
        return self._exchange

    @exchange.setter
    def exchange(self, exchange: 'Exchange'):
        raise ValueError("You cannot change a Wallet's Exchange after initialization.")

    @property
    def instrument(self) -> 'Instrument':
        return self._instrument

    @instrument.setter
    def instrument(self, instrument: 'Exchange'):
        raise ValueError("You cannot change a Wallet's Instrument after initialization.")

    @property
    def balance(self) -> 'Quantity':
        """The total balance of the wallet available for use."""
        return self._balance

    @balance.setter
    def balance(self, balance: 'Quantity'):
        self._balance = balance

    @property
    def locked_balance(self) -> 'Quantity':
        """The total balance of the wallet locked in orders."""
        locked_balance = Quantity(self.instrument, 0)

        for quantity in self.locked.values():
            locked_balance += quantity.size

        return locked_balance

    @property
    def total_balance(self) -> 'Quantity':
        """The total balance of the wallet, both available for use and locked in orders."""
        total_balance = self._balance

        for quantity in self.locked.values():
            total_balance += quantity.size

        return total_balance

    @property
    def locked(self) -> Dict[str, 'Quantity']:
        return self._locked

    def deallocate(self, path_id: str):
        if path_id in self.locked.keys():
            quantity = self.locked.pop(path_id, None)

            if quantity is not None:
                self += quantity.size * self.instrument

    #???????????????????????????Wallet?????????????????????????????????
    def __iadd__(self, quantity: 'Quantity') -> 'Wallet':
        if quantity.is_locked:
            if quantity.path_id not in self.locked.keys():
                self._locked[quantity.path_id] = quantity
            else:
                self._locked[quantity.path_id] += quantity
        else:
            self._balance += quantity

        return self

    def __isub__(self, quantity: 'Quantity') -> 'Wallet':
        if quantity.is_locked:
            if quantity.path_id in self.locked:
                if quantity > self.locked[quantity.path_id]:
                    raise InsufficientFunds(self.locked[quantity.path_id], quantity.size)
                self._locked[quantity.path_id] -= quantity
            else: 
                raise KeyError("locked quantity path id not found in wallet--> the substraction cannot be down! --Jianmin Mao")
        else:
            if quantity > self._balance:
                raise InsufficientFunds(self.balance, quantity.size)
            self._balance -= quantity
        return self

    def __str__(self):
        return '<Wallet: balance={}, locked={}>'.format(self.balance, self.locked_balance)

    def __repr__(self):
        return str(self)
     
    #from Adam King
    def reset(self):
        self._balance = Quantity(self._instrument, self._initial_size)
        self._locked = {}

