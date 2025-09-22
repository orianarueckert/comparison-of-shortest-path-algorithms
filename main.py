from short_path_finder import ShortPathFinder

from heuristic_graph import HeuristicGraph
from weighted_graph import WeightedGraph

from dijkstra import Dijkstra
from bellman_ford import BellmanFord
from a_star_adapted import AStarAdapted
from all_pairs_shortest_path import AllPairsSP

import random
import timeit
import matplotlib.pyplot as plt
import math 
import csv
import copy

# PART 2:
def draw_plot_part2(d_times, b_times, avg_d, avg_b):
    trials = list(range(1, len(d_times) + 1)) # Match the number of trials 

    # Plot the run times
    plt.plot(trials, d_times, marker='o', linestyle='-', color='blue', label="Dijkstra")
    plt.plot(trials, b_times, marker='o', linestyle='-', color='purple', label="Bellman Ford")

    # Plot the average run times
    plt.axhline(y=avg_d, color='blue', linestyle='dotted')
    plt.axhline(y=avg_b, color='purple', linestyle='dotted')

    # Labels and title
    plt.xlabel("Trials")
    plt.ylabel("Run Time (seconds)")
    plt.title("Dijkstra vs. Bellman Ford")

    # Show legend, grid and plot 
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to generate random graphs 
# We initialize min_weight to 1 because we must run Dijkstra and Bellman Ford on the same graphs (Dijkstra's only works for positive edge weights)
# A graphs density is the measure of (number of edges / maximum number of edges) 
    # Sparse graph -> density <= 0.1 (10%)
    # Moderate graph -> 0.1-0.5 density (11-49%)
    # Dense graph -> density >= 0.5 (50%+) 
    # Complete graph -> density = 1.0 (100%)
def create_random_graph_part2(nodes, density, min_weight=1, max_weight=10):
    # A graph cannot exist with zero nodes
    if nodes == 0: 
        return print("A graph with zero nodes cannot exist.")

    # Cannot have more edges than pairs of nodes possible
    max_edges = (nodes * (nodes - 1) // 2)
    num_edges = min(int(density * max_edges), max_edges) # Convert the given density into a number of edges
    
    graph = WeightedGraph()
    unique_edges = set() # Initialize a set to track which edges have been created and avoid duplicates 

    for i in range(nodes): 
        graph.add_node(i)

    while len(unique_edges) < num_edges: 
        # Generate two random nodes to connect with an edge 
        node1 = random.randint(0, nodes - 1)
        node2 = random.randint(0, nodes - 1)
        
        # Check if the edge already exists (self loops are not allowed)
        if (node1, node2) not in unique_edges and node1 != node2:
            weight = random.randint(min_weight, max_weight)
            unique_edges.add((node1, node2))
            graph.add_edge(node1, node2, weight) 

    return graph

# This experiment allows us to analyze the performance of Dijkstra and Bellman Ford for graphs of varying sizes, densities and k values 
# Takes in a list of sizes, densities and k values which are equal in length 
# Idea: use a varying set of sizes, densities or k values while keeping others relatively constant to observe how performance changes under different conditions
def shortest_path_experiment(sizes, densities, k_values): 
    # Experiment conditions 
    input_len = len(sizes)
    trials = 30 

    # Average run times for Dijkstra's and Bellman Ford algorithms over specified trial number 
    avg_d_times = []
    avg_b_times = []

    # Experiment conditions for each trial set 
    for i in range(input_len):
        size = sizes[i]
        density = densities[i]
        k = k_values[i] 
        source = 0 
        
        d_times_per_trial = [] 
        b_times_per_trial = [] 

        # Run the expirement for the specified number of trials
        for _ in range(trials):
            G = create_random_graph_part2(size, density) 

            test1 = ShortPathFinder()
            test1.set_graph(G)
            test1.set_algorithm(Dijkstra())

            # ----- Dijkstra's -----
            start = timeit.default_timer()
            test1.calc_short_path(source, None, k)
            stop = timeit.default_timer()
            d_times_per_trial.append(stop-start)

            # ----- Bellman Ford -----
            test2 = ShortPathFinder()
            test2.set_graph(G)
            test2.set_algorithm(BellmanFord())

            start = timeit.default_timer()
            test2.calc_short_path(source, None, k)
            stop = timeit.default_timer()
            b_times_per_trial.append(stop-start)

        # Average run times for this trial 
        avg_d_times.append(sum(d_times_per_trial) / len(d_times_per_trial))
        avg_b_times.append(sum(b_times_per_trial) / len(b_times_per_trial))

    # Overall average for all trials  
    overall_avg_d = (sum(avg_d_times) / len(avg_d_times))
    print("Average run time for Dijkstra's (seconds): ", overall_avg_d)

    overall_avg_b = (sum(avg_b_times) / len(avg_d_times))
    print("Average run time for Bellman Ford (seconds): ", overall_avg_b)

    # Plot the expiremental results for each algorithm 
    draw_plot_part2(avg_d_times, avg_b_times, overall_avg_d, overall_avg_b)

# ~ Must ensure the chosen k value is 1 < k < N-1 (where N is the number of nodes in the graph) ~ 

# Experiment test cases used in report 
shortest_path_experiment([20,20,20,20,20], [0.2,0.3,0.5,0.75,1], [15,15,15,15,15]) # Increasing density for small size and large k 
shortest_path_experiment([10,30,50,80,120], [0.1,0.1,0.1,0.1,0.1], [3,3,3,3,3]) # Increasing size for sparse graphs and small k 
shortest_path_experiment([40,40,40,40,40], [0.3,0.3,0.3,0.3,0.3], [5,10,20,30,38]) # Moderately sized graph with moderate density and increasing k 



# PART 3:
def create_random_graph_part3(nodes, edges):
    edge_pairs = []

    while edges > 0:
        # assuming nodes are always named 0 through (nodes - 1)
        random_node_1 = random.randint(0, nodes - 1)     
        random_node_2 = random.randint(0, nodes - 1)  
        
        # ensure that you are not adding parallel edges or self loops
        while [random_node_1,random_node_2] in edge_pairs or [random_node_2,random_node_1] in edge_pairs or random_node_1 == random_node_2:
            random_node_1 = random.randint(0, nodes - 1)     
            random_node_2 = random.randint(0, nodes - 1)  
    
        random_edge_weight = random.randint(-10, 20)
        
        # add edge between 2 randomly selected nodes
        edge_pairs.append([random_node_1, random_node_2, random_edge_weight])
        edges-=1
    return edge_pairs

def part_3_testing():
    # initialze a weighted graph
    all_pairs_test = WeightedGraph()
    edge_list = create_random_graph_part3(5, 10)
    for i in range(5):
        all_pairs_test.add_node(i)
    for pair in edge_list:
        all_pairs_test.add_edge(pair[0], pair[1], pair[2])

    # initialize a short path finder
    all_pairs_shortest_path_test = ShortPathFinder()
    all_pairs_shortest_path_test.set_graph(all_pairs_test)
    all_pairs_shortest_path_test.set_algorithm(AllPairsSP())

    print(all_pairs_test.adj)
    for src in all_pairs_test.adj:
        for dst in all_pairs_test.adj[src]:
            print("(", src, ",", dst, "):", all_pairs_test.w(src, dst))
            res_ = all_pairs_shortest_path_test.calc_short_path(src, dst, None)
            if res_ == [None, None]:
                break

    print("Distance Matrix:", all_pairs_shortest_path_test.get_cameFrom_path(None, None, None))
    

part_3_testing()
part_3_testing()
part_3_testing()
part_3_testing()


# PART 4
def part_4_testing():
    part_4_test_graph = HeuristicGraph()

    for i in range(5 + 1):
        part_4_test_graph.add_node(i)

    part_4_test_graph.add_edge(0, 1, 4)
    part_4_test_graph.add_edge(0, 5, 3)
    part_4_test_graph.add_edge(1, 3, 5)
    part_4_test_graph.add_edge(1, 4, 1)
    part_4_test_graph.add_edge(5, 2, 3)
    part_4_test_graph.add_edge(2, 0, 2)
    part_4_test_graph.add_edge(0, 2, 10)
        
    # heu is a function which provides an estimate for how far each node in the graph is from the dst node
    def heu(dst):
        return {node: float(abs(dst - node)) for node in range(part_4_test_graph.get_num_of_nodes())}

    heuristic = heu(2)  # heuristic from every node to 2
    print(heuristic)

    part_4_test_graph.set_heuristic(heuristic)

    part_4_test = ShortPathFinder()
    part_4_test.set_graph(part_4_test_graph)
    part_4_test.set_algorithm(AStarAdapted())

    print(part_4_test.calc_short_path(0, 2, None))    # should return 6 (since now we sum edge-weights)

part_4_testing()


# PART 5:
# Plot results for part 5
def draw_plot_part5(a_times, d_times, avg_a, avg_d, title):
    trials = list(range(1, len(a_times) + 1)) # Match the number of trials 

    plt.figure(figsize=(20, 6))

    bar_width = 0.35

    positions_a = [t - bar_width/2 for t in trials]
    positions_d = [t + bar_width/2 for t in trials]

    # Plot the run times
    plt.bar(positions_a, a_times, width=bar_width, color='red', label="A*")
    plt.bar(positions_d, d_times, width=bar_width, color='blue', label="Dijkstra")

    # Plot the average run times
    plt.axhline(y=avg_a, color='red', linestyle='dotted',label="A* Average")
    plt.axhline(y=avg_d, color='blue', linestyle='dotted',label="Dijkstras Average")

    # Labels and title
    plt.xlabel("Trials")
    plt.ylabel("Run Time (seconds)")
    plt.title(title)

    # Show legend, grid and plot 
    plt.legend()
    plt.grid(True)
    plt.show()


def EuclideanDistance(src_lat, src_lon, dst_lat, dst_lon):  # function to find distance between two coordinates
    return math.sqrt((dst_lat-src_lat)**2 + (dst_lon-src_lon)**2)  

# since AStar needs a heuristic graph while Dijkstras needs a weighted graph we will create 2
london_subway_system_H = HeuristicGraph()  # create a graph to represent the london subway system
london_subway_system = WeightedGraph()
station_location = {}  # initialize dictionary to store the locations of stations
lines = {}  # initialize dictionary to store the line that each connection is on (key is a tuple, value is the line)

with open("london_stations.csv", "r") as london_stations:  #  open london_stations.csv in read mode and name it london_stations
    dictionaries = csv.DictReader(london_stations)  # creates a dictionary for each row in the table, using the column name as a key

    for row in dictionaries:  # get each dictionary (which corresponds to a row in the table)
        station = int(row["id"])  # get the station id
        station_location[station] = [float(row["latitude"]), float(row["longitude"])]  # add the latitude and longitude as a list to the location dictionary

        london_subway_system.add_node(station)  # add the station to the graph as a node
        london_subway_system_H.add_node(station)

with open("london_connections.csv", "r") as london_connections:  #  open london_connections.csv in read mode and name it london_connections
    dictionaries = csv.DictReader(london_connections)  # creates a dictionary for each row in the table, using the column name as a key

    for row in dictionaries:  # get each dictionary (which corresponds to a row in the table)
        src = int(row["station1"])  # get the starting station
        dst = int(row["station2"])  # get the destination station
        distance = EuclideanDistance(station_location[src][0], station_location[src][1], station_location[dst][0], station_location[dst][1])  # compute the distance between the two coordinates

        london_subway_system.add_edge(src,dst,distance)  # add an edge from source to destination with weight
        london_subway_system.add_edge(dst,src,distance)  # add an edge from destination to source with weight (since lines run in both directions)
        london_subway_system_H.add_edge(src,dst,distance)  
        london_subway_system_H.add_edge(dst,src,distance)  


        lines[(src,dst)] = int(row["line"])  # record the line that the connection is on
        lines[(dst,src)] = int(row["line"])  # record the line that the connection is on (going the other way, since tuples are not ordered)

def h(dst):  # heuristic function which takes destination station as input and returns dictionary where keys are all other stations and values are the distance between them and dst
    dst_lat = station_location[dst][0]  # get the latitude of destination station
    dst_lon = station_location[dst][1]  # get the longitude of destination station
    dst_to_s = {}  # initialize an empty dictionary to store the estimated distance from dst to s

    for s in station_location:  # iterate through all the stations
        distance = EuclideanDistance(station_location[s][0], station_location[s][1], dst_lat, dst_lon)  # compute the distance between station s and the destination station
        dst_to_s[s] = distance  # add the station as a key and the distance of s to dst as the value
    
    return dst_to_s

# Part 5 Experiment Helper Functions
def line_count_AStar(path):  # function which counts how many lines are crossed in a given path found by A* from a src to dst station
    lines_crossed = set()  # initialize set to store the lines the path goes on

    for i in range(len(path)-1):  # iterate through all the stations on the path
        if (path[i],path[i+1]) in lines:
            lines_crossed.add(lines[(path[i],path[i+1])])  # add the line that each adjacent pair is on to the set
    
    return len(lines_crossed)  # return the number of lines that the path goes through

def line_count_Dijkstra(paths, dst):  # function which counts how many lines are crossed in a given path found by Dijkstra's from a src to dst station
    lines_crossed = set()  # initialize set to store the lines the path goes on
    path = paths[dst]  # get the path of the src to the dst
    
    for i in range(len(path)-1):  # iterate through all the stations on the path
        if (path[i],path[i+1]) in lines:
            lines_crossed.add(lines[(path[i],path[i+1])])  # add the line that each adjacent pair is on to the set
    
    return len(lines_crossed)  # return the number of lines that the path goes through

# experiment to find the shortest path between all station pairs
def experiment_part5():  

    # Initialize empty array to store the time each algorithm takes to find each path
    AStar_Everything = []
    Dijkstra_Everything = []

    # Initialize empty array to store the time each algorithm takes for each path (depending on how many lines that path goes through)
    AStar_SameLine = []
    AStar_AdjLines = []
    AStar_SeveralTransfers = []
    Dijkstra_SameLine = []
    Dijkstra_AdjLines = []
    Dijkstra_SeveralTransfers = []

    # nested for loop to go through every possible station pair
    for src in london_subway_system.adj:  
        for dst in london_subway_system.adj:
            london_subway_system_copy = copy.deepcopy(london_subway_system_H)
            london_subway_system_copy.set_heuristic(h(dst))
            res1 = ShortPathFinder()
            res1.set_graph(london_subway_system_copy)
            res1.set_algorithm(AStarAdapted())

            start = timeit.default_timer()
            result_A = res1.calc_short_path(src, dst, None)
            end = timeit.default_timer()

            AStar_Everything.append(end-start)  # add the time to the array storing execution times of A*

            result_A = res1.get_cameFrom_path(src, dst, None)

            # Depending on how many lines are transferred between the given src and dst, add the execution time of A* to the appropriate array
            if line_count_AStar(result_A[1]) == 1:
                AStar_SameLine.append(end-start)
            if line_count_AStar(result_A[1]) == 2:
                AStar_AdjLines.append(end-start)
            if line_count_AStar(result_A[1]) > 2:
                AStar_SeveralTransfers.append(end-start)
            
            # Time how long Dijkstra's takes
            london_subway_system_copy2 = copy.deepcopy(london_subway_system)
            res2 = ShortPathFinder()
            res2.set_graph(london_subway_system_copy2)
            res2.set_algorithm(Dijkstra())

            start = timeit.default_timer()
            result_D = res2.calc_short_path(src, dst, 30)  # pick any k such that k >= 1
            end = timeit.default_timer()

            Dijkstra_Everything.append(end-start)  # add the time to the array storing execution times of A*

            result_D = res2.get_cameFrom_path(src, dst, 30)
            # Depending on how many lines are transferred between the given src and dst, add the execution time of A* to the appropriate array
            if line_count_Dijkstra(result_D[1],dst) == 1:
                Dijkstra_SameLine.append(end-start)
            if line_count_Dijkstra(result_D[1],dst) == 2:
                Dijkstra_AdjLines.append(end-start)
            if line_count_Dijkstra(result_D[1],dst) > 2:
                Dijkstra_SeveralTransfers.append(end-start)

    AStar_mean = (sum(AStar_Everything)) / len(AStar_Everything)
    print("A* mean: " + str(AStar_mean))
    Dijkstra_mean = (sum(Dijkstra_Everything)) / len(Dijkstra_Everything)
    print("Dijkstra's mean: " + str(Dijkstra_mean))
    draw_plot_part5(AStar_Everything, Dijkstra_Everything, AStar_mean, Dijkstra_mean, "Finding Shortest Path Between Two Stations: A* vs. Dijkstra's")

    AStar_mean = (sum(AStar_SameLine)) / len(AStar_SameLine)
    print("A* mean: " + str(AStar_mean))
    Dijkstra_mean = (sum(Dijkstra_SameLine)) / len(Dijkstra_SameLine)
    print("Dijkstra's mean: " + str(Dijkstra_mean))
    draw_plot_part5(AStar_SameLine, Dijkstra_SameLine, AStar_mean, Dijkstra_mean, "Finding Shortest Path Between Two Stations on the Same Line: A* vs. Dijkstra's")

    AStar_mean = (sum(AStar_AdjLines)) / len(AStar_AdjLines)
    print("A* mean: " + str(AStar_mean))
    Dijkstra_mean = (sum(Dijkstra_AdjLines)) / len(Dijkstra_AdjLines)
    print("Dijkstra's mean: " + str(Dijkstra_mean))
    draw_plot_part5(AStar_AdjLines, Dijkstra_AdjLines, AStar_mean, Dijkstra_mean, "Finding Shortest Path Between Two Stations on Adjacent Lines: A* vs. Dijkstra's")

    AStar_mean = (sum(AStar_SeveralTransfers)) / len(AStar_SeveralTransfers)
    print("A* mean: " + str(AStar_mean))
    Dijkstra_mean = (sum(Dijkstra_SeveralTransfers)) / len(Dijkstra_SeveralTransfers)
    print("Dijkstra's mean: " + str(Dijkstra_mean))
    draw_plot_part5(AStar_SeveralTransfers, Dijkstra_SeveralTransfers, AStar_mean, Dijkstra_mean, "Finding Shortest Path Between Two Stations Needing Several Transfers: A* vs. Dijkstra's")

# experiment to pick random src and dst that require several transfers to get from src to dst and print the number of lines crossed
def num_lines_experiment_part5():  

    S = london_subway_system.get_num_of_nodes()

    while True:  # iterate until we find a src and dst that crosses several lines
        # get random stations
        src = random.choice(list(station_location.keys()))
        dst = random.choice(list(station_location.keys())) 

        london_subway_system_copy = copy.deepcopy(london_subway_system_H)
        london_subway_system_copy.set_heuristic(h(dst))
        res3 = ShortPathFinder()
        res3.set_graph(london_subway_system_copy)
        res3.set_algorithm(AStarAdapted())

        result = res3.get_cameFrom_path(src, dst, None) # compute the shortest path between src and dst using A*

        num_lines = line_count_AStar(result[1])  # get the number of lines the path uses
    
        if num_lines > 2:  # if the shortest path uses several lines...
            print("Shortest path between two random stations (requiring several transfers):",result[1])
            print("Number of lines crossed:",num_lines)
            break 

experiment_part5()
num_lines_experiment_part5()
num_lines_experiment_part5()
num_lines_experiment_part5()
num_lines_experiment_part5()
num_lines_experiment_part5()