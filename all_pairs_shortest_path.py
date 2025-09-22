from sp_algorithm import SPAlgorithm
from weighted_graph import WeightedGraph

import math

class AllPairsSP(SPAlgorithm):
    def __init__(self) -> None:
        super().__init__()
        self.dist =[[]]
        self.prev = [[]]

    def calc_sp(self, graph: WeightedGraph, source: int, dest: int, k: int = None) -> float:
        n = graph.get_num_of_nodes() 

        # data structures to hold weight from pairs of sources and dests as well as the order in the path
        self.dist = [[math.inf] * n for _ in range(n)]  
        self.prev = [[None] * n for _ in range(n)]
        
        # distance from a node to itself is 0
        for i in range(n):
            self.dist[i][i] = 0
        
        # add direct edge weights to data structures
        weights = graph.weights
        for src, dst in weights:
            self.dist[src][dst] = graph.weights[(src, dst)]
            self.prev[src][dst] = src
        
        # Floyd-Warshall algorithm
        for k in range(n):
            for src in range(n): 
                for dst in range(n):
                    if self.dist[src][k] + self.dist[k][dst] < self.dist[src][dst]:
                        self.dist[src][dst] = self.dist[src][k] + self.dist[k][dst]
                        self.prev[src][dst] = self.prev[k][dst]

        # check for negative cycles that will cause an incorrect shortest path
        for i in range(n):
            if self.dist[i][i] < 0:
                print("Negative weighted cycle detected!")
                return None, None

        return self.dist[source][dest]
    
    def get_dest(self,):
        return self.dist

    def get_prev(self,):
        return self.prev
    



