import operator

from typing import Union, Tuple
from numbers import Number

from tensortrade.base.exceptions import InvalidNegativeQuantity, IncompatibleInstrumentOperation, \
    InvalidNonNumericQuantity, QuantityOpPathMismatch


class Quantity:
    """An size of a financial instrument for use in trading.
    """

    def __init__(self, instrument: 'Instrument', size: float = 0, path_id: str = None):
        if size < 0:
            raise InvalidNegativeQuantity(size)

        self._size = size
        self._instrument = instrument
        self._path_id = path_id

    @property
    def size(self) -> float:
        return self._size

    @size.setter
    def size(self, size: float):
        self._size = size

    @property
    def instrument(self) -> 'Instrument':
        return self._instrument

    @instrument.setter
    def instrument(self, instrument: 'Exchange'):
        raise ValueError("You cannot change a Quantity's Instrument after initialization.")

    @property
    def path_id(self) -> str:
        return self._path_id

    @path_id.setter
    def path_id(self, path_id: str):
        self._path_id = path_id

    @property
    def is_locked(self) -> bool:
        return bool(self._path_id)

    def lock_for(self, path_id: str):
        self._path_id = path_id

    def free(self):
        return Quantity(self.instrument, self.size)

    @staticmethod
    def validate(left, right) -> Tuple['Quantity', 'Quantity']:

        if isinstance(left, Quantity) and isinstance(right, Quantity):
            if left.instrument != right.instrument:
                raise IncompatibleInstrumentOperation(left, right)

            if (left.path_id and right.path_id) and (left.path_id != right.path_id):
                raise QuantityOpPathMismatch(left.path_id, right.path_id)

            elif left.path_id and not right.path_id:
                right.path_id = left.path_id

            elif not left.path_id and right.path_id:
                left.path_id = right.path_id

            return left, right

        elif isinstance(left, Number) and isinstance(right, Quantity):
            left = Quantity(right.instrument, float(left), right.path_id)
            return left, right

        elif isinstance(left, Quantity) and isinstance(right, Number):
            right = Quantity(left.instrument, float(right), left.path_id)
            return left, right

        elif isinstance(left, Quantity):
            raise InvalidNonNumericQuantity(right)

        elif isinstance(right, Quantity):
            raise InvalidNonNumericQuantity(left)

        return left, right

    @staticmethod
    def _bool_operation(left: Union['Quantity', float, int],
                        right: Union['Quantity', float, int],
                        bool_op: operator) -> bool:
        left, right = Quantity.validate(left, right)

        boolean = bool_op(left.size, right.size)

        if not isinstance(boolean, bool):
            raise Exception("`bool_op` cannot return a non-bool type ({}).".format(boolean))

        return boolean

    @staticmethod
    def _math_operation(left: Union['Quantity', float, int],
                        right: Union['Quantity', float, int],
                        op: operator) -> 'Quantity':
        left, right = Quantity.validate(left, right)

        size = op(left._size, right._size)
        return Quantity(left.instrument, size, left.path_id)

    def __add__(self, other: Union['Quantity', float, int]) -> 'Quantity':
        return Quantity._math_operation(self, other, operator.add)

    def __sub__(self, other: Union['Quantity', float, int]) -> 'Quantity':
        return Quantity._math_operation(self, other, operator.sub)

    def __iadd__(self, other: Union['Quantity', float, int]) -> 'Quantity':
        return Quantity._math_operation(self, other, operator.iadd)

    def __isub__(self, other: Union['Quantity', float, int]) -> 'Quantity':
        return Quantity._math_operation(self, other, operator.isub)

    def __mul__(self, other: Union['Quantity', float, int]) -> 'Quantity':
        return Quantity._math_operation(self, other, operator.mul)

    def __rmul__(self, other: Union['Quantity', float, int]) -> 'Quantity':
        return Quantity.__mul__(self, other)

    def __lt__(self, other: Union['Quantity', float, int]) -> bool:
        return Quantity._bool_operation(self, other, operator.lt)

    def __gt__(self, other: Union['Quantity', float, int]) -> bool:
        return Quantity._bool_operation(self, other, operator.gt)

    def __eq__(self, other: Union['Quantity', float, int]) -> bool:
        return Quantity._bool_operation(self, other, operator.eq)

    def __ne__(self, other: Union['Quantity', float, int]) -> bool:
        return Quantity._bool_operation(self, other, operator.ne)

    def __neg__(self) -> bool:
        return operator.neg(self.size)

    def __str__(self):
        s = "{0:." + str(self.instrument.precision) + "f}" + " {1}"
        s = s.format(self.size, self.instrument.symbol)
        return s

    def __repr__(self):
        return str(self)
