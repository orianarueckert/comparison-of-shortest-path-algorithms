from weighted_graph import WeightedGraph

class HeuristicGraph(WeightedGraph):

    def __init__(self,):
        super().__init__()
        self._heuristic = {}  

    def get_heuristic(self, ) -> dict[int, float]:
        return self._heuristic
    
    # since we want this heuristic graph to work for whatever type of heuristic basis, the actual heuristic estimate is defined outside of this class
    # we also want to set the heuristic after the heuristic graph has been instantiated, so being able to set it later is necessary
    def set_heuristic(self, heuristic : dict[int, float]) -> dict[int, float]:
        self._heuristic = heuristic
    