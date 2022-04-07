import functools

from typing import Union, Callable

from .node import Node, Module


class BinOp(Node):

    def __init__(self, name: str, op):
        super().__init__(name)
        self.op = op

    def forward(self):  #原来用_value, 但有些node没有_value，并且基于forward的机制，此出复合node（Binop）在forward的时候，所有子node都应该进行forward操作
        return self.op(self.inputs[0].forward(), self.inputs[1].forward())

    def has_next(self):
        return True

    def reset(self):
        pass


class Reduce(Node):

    def __init__(self,
                 name: str,
                 func: Callable[[float, float], float]):
        super().__init__(name)
        self.func = func

    def forward(self):
        return functools.reduce(self.func, [node.forward() for node in self.inputs])

    def has_next(self):
        return True

    def reset(self):
        pass




class Lambda(Node):

    def __init__(self, name: str, extract: Callable[[any], float], obj: any):
        super().__init__(name)
        self.extract = extract
        self.obj = obj

    def forward(self):
        return self.extract(self.obj)

    def has_next(self):
        return True

    def reset(self):
        pass

"""exchange里面的_prices中的node是forward(stream),这个stream是价格，env.step的时候会被forward()，
  所以此处的forward node不会再次forward(),而是直接返回stream当前的value
"""
#static node in terms of using forward(), but if it's included in datafeed, then datafeed.next() will still update the nodes related to it if the nodes are also in its inputs
class Forward(Lambda):

    def __init__(self, node: 'Node'):
        super().__init__(
            name=node.name,
            extract=lambda x: x.value,
            obj=node
        )
        self(node)


class Condition(Module):

    def __init__(self, name: str, condition: Callable[['Node'], bool]):
        super().__init__(name)
        self.condition = condition

    def build(self):  #Module的flatten改成用inputs的，所以这里原来赋值给variables,改成赋值给inputs
        self.inputs = list(filter(self.condition, self.inputs))
        self.built = True

    def has_next(self):
        return True