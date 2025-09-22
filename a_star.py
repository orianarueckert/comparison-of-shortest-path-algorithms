import heapq

from heuristic_graph import HeuristicGraph

# this is the AStar algorithm that needs to be adapted since its return type is not compatible with the SPAlgorithm interface defined in the UML
class AStar():
    def calc_sp(self, graph : HeuristicGraph, source: int, dest : int) -> tuple[dict[int, int], list[int]]:
        openSet = [] # initialize priority queue

        h = graph.get_heuristic()

        heapq.heappush(openSet, (0 + h[source], source))  # push src node and its fScore into the priority queue (heappush is used to maintain the heap property)
        
        cameFrom = {}  # initialize a dictionary where cameFrom[n] is the node preceding node n on the cheapest path from the start
    
        # gScore is the cost of getting from src to any node
        gScore = {n: float('inf') for n in graph.adj}  # initialize gScore as inifinity for each node n
        gScore[source] = 0  # set the cost of getting from src to src as 0
        
        # fScore is the current best guess for the cheapest path from src to dst through node n
        # fScore(n) = gScore(n) + h(n)
        fScore = {n: float('inf') for n in graph.adj}  # initialize fScore as inifinity for each node n
        fScore[source] = h[source]  # initialize fScore for src node
        
        while openSet != []:  # while the open set is not empty
            fScore_value, current = heapq.heappop(openSet)  # remove the node with the smallest fScore from the open set and call it 'current'
            
            if current == dest:  # if the node we removed is the dst, we want to reconstruct the path
                path = []  # initialize an empty array to store the path
                while current in cameFrom:  # work backwards from dst to src...
                    path.append(current)  # append 'current' to the path
                    current = cameFrom[current]  # set 'current' to be the predecessor of itself
                path.append(source)  # add src to the path
                path.reverse()  # reverse the path (so that it goes from src to dst)
                return cameFrom, path  # return the predecessor dictionary and the shortest path from src to dst
            
            for neighbour in graph.get_adj_nodes(current):  # iterate over nodes adjacent to 'current'
                tentative_gScore = gScore[current] + graph.w(current, neighbour)  # tentative gScore is distance from src to neighbour through 'current'
                
                if tentative_gScore < gScore[neighbour]:  # if this path to neighbour is better than a previous one, record it
                    cameFrom[neighbour] = current  # update the predecessor of neighbour to 'current'
                    gScore[neighbour] = tentative_gScore  # update the cost to reach the neighbour
                    fScore[neighbour] = tentative_gScore + h[neighbour]  # update the best guess for the cheapest path
                    heapq.heappush(openSet, (fScore[neighbour], neighbour))  # push the neighbour and its fScore into the priority queue
        
        return cameFrom, []  # no path found so return the predecessor dictionary and an empty path