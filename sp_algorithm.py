from graph import Graph

class SPAlgorithm():
    def calc_sp(self, graph : Graph, source: int, dest : int, k : int) -> float:
        """
        Finds the shortest path between source and destination in given graph.
        Since in nowhere in this project we are finding paths from source to destination, I took out dest as a parameter to simplify things as making it an optional parameter would cause issues in other places in the code. 

        params:
            graph: The graph on which to run the algorithm
            source: The starting node
            k : Max number of times a node can be relaxed where 0 < k < N - 1 such that N is number of nodes in the graph
        return 
            list that holds the minimum edge-weights to get from source to destination
            list that holds the previous node on the shortest path from a node at the ith index 
        """
        pass
