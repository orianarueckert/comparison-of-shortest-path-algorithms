from sp_algorithm import SPAlgorithm
from weighted_graph import WeightedGraph

class Dijkstra(SPAlgorithm):
    def __init__(self) -> None:
        self.dist = {}
        self.path = {}

    def calc_sp(self, graph : WeightedGraph, source: int, dest : int, k : int) -> float:
        min_heap = MinHeap([])
        relaxed = {} # Number of times each node has been relaxed
        in_heap = {} 

        # For every node in the graph, initialize its distance from the source, its previous node and the number of times it has been relaxed 
        for node in graph.adj: 
            self.dist[node] = float('inf')
            self.path[node] = []
            relaxed[node] = 0 
            in_heap[node] = True 
            min_heap.insert(Item(node, self.dist[node])) # Add each node to the min heap with the distance as the key

        # Initialize the distance to the source node as zero, update the min heap and path  
        min_heap.decrease_key(source, 0)
        self.dist[source] = 0 
        self.path[source] = [source]

        # Iterate through every node in the min heap 
        while not min_heap.is_empty(): 
            smallest = min_heap.extract_min().value # Extract the node with the smallest distance
            in_heap[smallest] = False # Extracting a node removes it from the heap 

            if dest is not None and smallest == dest:
                return self.dist[dest]

            for neighbour in graph.adj[smallest]: 
                if in_heap[neighbour] == True: 
                    alt_dist = self.dist[smallest] + graph.w(smallest, neighbour) # Calculate the alternate distance 

                    # If the alternate distance is less than the existing distance: 
                    # Update the shortest distance, previous node, number of relaxations and min heap   
                    if alt_dist < self.dist[neighbour] and relaxed[neighbour] < k:
                        self.dist[neighbour] = alt_dist
                        self.path[neighbour] = self.path[smallest] + [neighbour] # Build the shortest path 
                        relaxed[neighbour] += 1 # Increment relaxation count 
                        min_heap.decrease_key(neighbour, alt_dist)

        # In part 2 we want Dijkstras to "return a distance and path dictionary". This contradicts the return type in the UML but is necessary when we are not given a destination
        if dest is None:
            return self.get_dist(), self.get_path()
        
        return self.dist[dest]
    
    def get_dist(self) -> dict:
        return self.dist
    
    def get_path(self) -> dict:
        return self.path

"""
"Helper" classes for Dijkstras. 
We decided not to use heapq for this implementation so that we could use the decrease_key() function to improve runtime and make our operations more clear. 
"""
class Item:
    def __init__(self, value, key):
        self.key = key
        self.value = value


class MinHeap():
    def __init__(self, data):
        self.items = data
        self.length = len(data)
        self.build_heap()

        # Map each node value to its respective index (Ex. {A : [1]})
        self.map = {}
        for i in range(self.length):
            self.map[self.items[i].value] = i

    def find_left_index(self,index):
        return 2 * (index + 1) - 1

    def find_right_index(self,index):
        return 2 * (index + 1)

    def find_parent_index(self,index):
        return (index + 1) // 2 - 1  
    
    def heapify(self, index):
        smallest_known_index = index

        if self.find_left_index(index) < self.length and self.items[self.find_left_index(index)].key < self.items[index].key:
            smallest_known_index = self.find_left_index(index)

        if self.find_right_index(index) < self.length and self.items[self.find_right_index(index)].key < self.items[smallest_known_index].key:
            smallest_known_index = self.find_right_index(index)

        if smallest_known_index != index:
            self.items[index], self.items[smallest_known_index] = self.items[smallest_known_index], self.items[index]
            
            # Update map
            self.map[self.items[index].value] = index
            self.map[self.items[smallest_known_index].value] = smallest_known_index

            # Recursive call
            self.heapify(smallest_known_index)

    def build_heap(self,):
        for i in range(self.length // 2 - 1, -1, -1):
            self.heapify(i) 

    def insert(self, node):
        if len(self.items) == self.length:
            self.items.append(node)
        else:
            self.items[self.length] = node

        self.map[node.value] = self.length
        self.length += 1
        self.swim_up(self.length - 1)

    def insert_nodes(self, node_list):
        for node in node_list:
            self.insert(node)

    def swim_up(self, index):
        while index > 0 and self.items[index].key < self.items[self.find_parent_index(index)].key:
            # Swap values
            self.items[index], self.items[self.find_parent_index(index)] = self.items[self.find_parent_index(index)], self.items[index]
            
            # Update map
            self.map[self.items[index].value] = index
            self.map[self.items[self.find_parent_index(index)].value] = self.find_parent_index(index)
            
            # Recursive call 
            index = self.find_parent_index(index)

    def get_min(self):
        if len(self.items) > 0:
            return self.items[0]

    def extract_min(self,):
        # Exchange
        self.items[0], self.items[self.length - 1] = self.items[self.length - 1], self.items[0]

        # Update map
        self.map[self.items[self.length - 1].value] = self.length - 1
        self.map[self.items[0].value] = 0

        min_node = self.items[self.length - 1]
        self.length -= 1
        self.map.pop(min_node.value)
        self.heapify(0)

        return min_node
    
    def decrease_key(self, value, new_key):
        if new_key >= self.items[self.map[value]].key:
            return
        index = self.map[value]
        self.items[index].key = new_key
        self.swim_up(index)

    def get_element_from_value(self, value):
        return self.items[self.map[value]]

    def is_empty(self):
        return self.length == 0