from typing import List
from tensortrade.data.stream.node import Node


class Stream(Node):

    def __init__(self, name: str, array: List[any] = None):
        super().__init__(name)
        self._array = array if array else []
        self._cursor = 0
        self._node_type = "Stream"

    def forward(self):
        self._value = self._array[self.cursor]
        self._cursor += 1
        return self.value

    def has_next(self) -> bool:
        if self.cursor < len(self):
            return True
        return False

    def reset(self):
        self._cursor = 0
        
    @property
    def cursor(self):
        return self._cursor
    
    def __len__(self):
        return len(self._array)
