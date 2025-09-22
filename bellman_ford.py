from sp_algorithm import SPAlgorithm
from weighted_graph import WeightedGraph
from typing import Union

class BellmanFord(SPAlgorithm):
    def __init__(self) -> None:
        self.dist = {}
        self.path = {}
    
    def calc_sp(self, graph : WeightedGraph, source: int, dest : int, k : int):
        # For every node in the graph, initialize its distance from the source and its previous node 
        for node in graph.adj: 
            self.dist[node] = float('inf')
            self.path[node] = []

        # Initialize the distance to the source node as zero 
        self.dist[source] = 0 
        self.path[source]  = [source] 

        # Relax all edges k times 
        for _ in range(k): 
            for src in graph.adj:
                for dst in graph.adj[src]:
                    alt_dist = self.dist[src] + graph.w(src, dst) # Calculate the alternate distance 

                    # If the alternate distance is less than the existing distance: 
                    # Update the shortest distance, previous node  
                    if alt_dist < self.dist[dst]:
                        self.dist[dst] = alt_dist
                        self.path[dst] = self.path[src] + [dst]

            # path to destination has been found so we can exit early and return
            if dest is not None and self.dist[dest] < float('inf'):
                break
        
        # In part 2 we want Bellman-Ford to "return a distance and path dictionary". This contradicts the return type in the UML but is necessary when we are not given a destination.
        if dest is None:
            return self.get_dist(), self.get_path()
        
        return self.dist[dest]
    
    def get_dist(self) -> dict:
        return self.dist
    
    def get_path(self) -> dict:
        return self.path