from heuristic_graph import HeuristicGraph
from sp_algorithm import SPAlgorithm
from a_star import AStar

class AStarAdapted(SPAlgorithm):
    def __init__(self) -> None:
        self.adaptee_AStar = AStar()

    def calc_sp(self, graph: HeuristicGraph, source: int, dest: int, k=None) -> float:
        cameFrom, path = self.adaptee_AStar.calc_sp(graph, source, dest)

        # sum edge-weights along path from source to dest to align with UML
        cost = 0
        for i in range(len(path) - 1):
            cost += graph.w(path[i], path[i+1])
        return cost
    
    def get_cameFrom_path(self, graph: HeuristicGraph, source : int, dest : int, k = None) -> tuple[dict[int, int], list[int]]:
        return self.adaptee_AStar.calc_sp(graph, source, dest)
    