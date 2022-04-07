
from typing import List

from tensortrade.data.stream import Node


class DataFeed(Node):

    def __init__(self, nodes: List[Node] = None):
        super().__init__("")

        self.queue = None
        self.compiled = False

        if nodes:
            self.__call__(*nodes)

        self._max_len = 0
        
        self._node_type = "DataFeed"
        
        self.compile()

    @property
    def max_len(self):
        return self._max_len
    
    @staticmethod
    def _gather(node, vertices, edges):
        if node not in vertices:
            vertices += [node]

            for input_node in node.inputs:
                edges += [(input_node, node)]

            for input_node in node.inputs:
                DataFeed._gather(input_node, vertices, edges)

        return edges

    def gather(self):
        return self._gather(self, [], [])

    @staticmethod
    def toposort(edges):
        S = set([s for s, t in edges])
        T = set([t for s, t in edges])

        starting = list(S.difference(T))
        queue = starting.copy()

        while len(starting) > 0:
            start = starting.pop()

            edges = list(filter(lambda e: e[0] != start, edges))

            S = set([s for s, t in edges])
            T = set([t for s, t in edges])

            starting += [v for v in S.difference(T) if v not in starting]

            if start not in queue:
                queue += [start]

        return queue

    def compile(self):
        edges = self.gather()

        self.queue = self.toposort(edges)
        
        for node in self.inputs:
            self._max_len = max(self.max_len, len(node))
            
        self._streams = []
        self._non_streams = []
        for node in self.queue:
            if node.node_type == "Stream":
                self._streams.append(node)
            else:
                self._non_streams.append(node)
        self.compiled = True
        self.reset()

    def run(self):

        for node in self._streams:
            node.run()  #此处的node，如果是externel feed, 一般是Stream, Stream 在forward时会涉及时间（Stream.run()会让时间流动，数据更新）
            #node.run(): node.value = node.forward()
        for node in self._non_streams:  #先更新Streams，再更新非Streams，原因是这些非Streams的值有可能depend on Streams
            node.run()
        super().run()  #datafeed.value = node.forward ——>datafeed.forward ->dict with all values of the nodes in datafeed

    def forward(self):
        return {node.name: node.value for node in self.inputs}

    def next(self, steps: int = 1):
        for _ in range(steps):
            self.run()
    
            #after the feed has been attached by some portfolios, it has listeners
            for listener in self.listeners:
                listener.on_next(self.value)

        return self.value

    def has_next(self) -> bool:
        return all(node.has_next() for node in self.queue)

    def __add__(self, other):
      ##From Adam King
        if not other.node_type == "DataFeed":
            raise TypeError(f'can only concatenate DataFeed (not "{type(other).__name__}") to DataFeed.')

        nodes = self.inputs + other.inputs
        feed = DataFeed(nodes)

        for listener in self.listeners + other.listeners:
            feed.attach(listener)

        return feed

    def reset(self):
        for node in self.queue:
            node.reset()
