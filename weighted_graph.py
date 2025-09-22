from graph import Graph

class WeightedGraph(Graph):
    def __init__(self,) -> None:
        super().__init__()
        self.weights = {}  

    def get_adj_nodes(self, node: int) -> list[int]:
        return self.adj[node]

    def add_node(self, node: int):
        self.adj[node] = []

    def add_edge(self, start: int, end: int, w: float):
        if end not in self.adj[start]:
            self.adj[start].append(end)
        self.weights[(start, end)] = w

    def get_num_of_nodes(self) -> int:
        return len(self.adj)

    def w(self, node1: int, node2: int) -> float:
        for neighbour in self.adj[node1]:
            if neighbour == node2:
                return self.weights[(node1, node2)]
        return float('inf')