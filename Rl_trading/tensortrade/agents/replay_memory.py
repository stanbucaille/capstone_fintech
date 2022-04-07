
import random

from collections import namedtuple
from typing import List


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'done'])


class ReplayMemory(object):

    def __init__(self, capacity: int, transition_type: namedtuple = Transition):
        self._capacity = capacity
        self.Transition = transition_type

        self._memory = []
        self.position = 0

    @property
    def memory(self):
        return self._memory
    
    @memory.setter
    def memory(self, memo):
        if len(memo)>self._capacity:
            raise ValueError("size of the memory is larger than the capacity of the ReplayMemory object!!  --Jianmin Mao")
        self._memory = memo
        
    @property
    def capacity(self):
        return self._capacity
    
    
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(self.Transition(*args))
        else:
            self.memory[self.position] = self.Transition(*args)

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size) -> List[namedtuple]:
        if batch_size > self.capacity:
            raise NotImplementedError("batch size is larger than memory's capacity!! --Jianmin Mao")
        return random.sample(self.memory, batch_size)

    def head(self, batch_size) -> List[namedtuple]:
        if batch_size > self.capacity:
            raise NotImplementedError("batch size is larger than memory's capacity!! --Jianmin Mao")
        return self.memory[:batch_size]

    def tail(self, batch_size) -> List[namedtuple]:
        if batch_size > self.capacity:
            raise NotImplementedError("batch size is larger than memory's capacity!! --Jianmin Mao")
        return self.memory[-batch_size:]

    def __len__(self):
        return len(self.memory)
