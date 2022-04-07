"""
References:
    - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/module/module.py
    - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/base_layer.py
    - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/node.py
"""

from abc import abstractmethod
from tensortrade.base.core import Observable
#from .transform import Select  #如果引用子类,子类文件里又引用父类，循环引用
from typing import Union, Callable

class Node(Observable):

    def __init__(self, name: str):
        super().__init__()
        self._node_type = None
        self._name = name
        self.inputs = []

        for i in range(len(Module.CONTEXTS)):
            Module.CONTEXTS[i].add_node(self) 

    @property
    def name(self):
        return self._name
    
    @property
    def node_type(self):
        return self._node_type

    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value: float):
        self._value = value

    def __call__(self, *inputs):  #involve other nodes into current node
        
        for node in inputs:
            if node in self.inputs:
                pass
            elif isinstance(node, Module) or hasattr(node, 'flatten'):  #if this node is a module, then flatten it to get its components(nodes)
                if not node.built:
                    node.build()
                    node.built = True
                flatten_nodes = node.flatten()  
                    
                self.inputs += flatten_nodes  
            else:
                self.inputs += [node]

        return self

    def run(self):
        self.value = self.forward()

    @abstractmethod
    def forward(self):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def has_next(self):
        raise NotImplementedError()

    def __str__(self):
        return "<Node: name={}, type={}>".format(self.name,
                                                 str(self.__class__.__name__).lower())

    def __repr__(self):
        return str(self)

    def __len__(self):
        return 0

#currently, exhange, condition are modules. externel feed is also constructed using module
class Module(Node):

    CONTEXTS = []

    def __init__(self, name: str):
        super().__init__(name)

        self.submodules = []
        self.built = False
        self._node_type = "Module"

    def add_node(self, node: 'Node'):
        node.name = self.name + ":/" + node.name

        if isinstance(node, Module) or hasattr(node, 'flatten'):
            self.submodules += [node]
        else:
            self.inputs += [node]
			
    @abstractmethod
    def build(self): 
        pass

    def flatten(self):
        nodes = [node for node in self.inputs]

        for module in self.submodules:
            nodes += module.flatten()

        return nodes

    def __enter__(self):
        Module.CONTEXTS += [self]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Module.CONTEXTS.pop()
        return self

    def forward(self):
        return

    def reset(self):
        pass

#static node
#select the first one in its inputs with specified name
class Select(Node):

    def __init__(self, selector: Union[Callable[[str], bool], str], key: str=None):
        self.key = key
        if isinstance(selector, str):
            self.selector = lambda x: x.name == selector
        else:
            self.selector = selector
        super().__init__(self.key or "select")
        self._node = None
        self._node_type = "Select"

    def forward(self):
        if not self._node:
            self._node = list(filter(self.selector, self.inputs))[0]
            self.name = self._node.name
        return self._node.value

    def has_next(self):
        return True

    def reset(self):
        pass