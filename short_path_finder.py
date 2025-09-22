from graph import Graph
from sp_algorithm import SPAlgorithm
from a_star_adapted import AStarAdapted
from dijkstra import Dijkstra
from bellman_ford import BellmanFord
from all_pairs_shortest_path import AllPairsSP

class ShortPathFinder:
    def __init__(self):
        # we will set the values of these attributes later
        self.graph = None
        self.algorithm = None

    def set_graph(self, graph: Graph):
        self.graph = graph

    def set_algorithm(self, algorithm: SPAlgorithm):
        self.algorithm = algorithm

    def calc_short_path(self, source: int, dest: int, k: int) -> float:
        return self.algorithm.calc_sp(self.graph, source, dest, k)
    
    def get_cameFrom_path(self, source: int, dest: int, k: int) -> tuple[dict[int, float], dict[int, list[int]]]:
        if isinstance(self.algorithm, AStarAdapted):
            return self.algorithm.get_cameFrom_path(self.graph, source, dest, k)
        
        if isinstance(self.algorithm, Dijkstra):
            return self.algorithm.get_dist(), self.algorithm.get_path()
        
        if isinstance(self.algorithm, BellmanFord):
            return self.algorithm.get_dist(), self.algorithm.get_path()
        
        if isinstance(self.algorithm, AllPairsSP):
            return self.algorithm.get_dest(), self.algorithm.get_prev()
        
        return None