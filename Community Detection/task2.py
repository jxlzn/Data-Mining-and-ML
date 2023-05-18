import json
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark import SparkContext
import os
import csv
from itertools import combinations
import sys
import time
import random
from operator import add
import operator
from math import sqrt
from collections import defaultdict

sys.setrecursionlimit(20000)

#os.environ['PYSPARK_PYTHON'] = 'C:/Users/jxlzn/AppData/Local/Programs/Python/Python36/python.exe'
#os.environ['PYSPARK_DRIVER_PYTHON'] = 'C:/Users/jxlzn/AppData/Local/Programs/Python/Python36/python.exe'

start = time.time()


spark = SparkSession.builder.appName("task1").getOrCreate()
sc = SparkContext.getOrCreate()

sc.setLogLevel('WARN')

file_path = sys.argv[2]

data_rdd = sc.textFile(file_path)



header = data_rdd.first()

data = data_rdd.filter(lambda x: x != header).map(lambda x: x.split(","))

user_business_list = data.map(lambda x: (x[0], x[1])).groupByKey().mapValues(lambda x: set(x))

# print(userlist.collect())


threshold = int(sys.argv[1])

def common_pairs(user_pair, threshold):
    user_i = user_pair[0]
    user_j = user_pair[1]

    common_businesses = user_i[1].intersection(user_j[1])

    if len(common_businesses) >= int(threshold):
        #return [(user_i[0], user_j[0]), (user_j[0], user_i[0])]
        return [(user_i[0], user_j[0])]
    else:
        return []


def create_edges(user_business_list, threshold):

    user_pairs = user_business_list.cartesian(user_business_list).filter(lambda x: x[0][0] < x[1][0])


    edges = user_pairs.flatMap(lambda user_pair: common_pairs(user_pair, threshold))
    return edges


edges = create_edges(user_business_list, threshold)

distinct_edges_rdd = edges.distinct()

distinct_edges = distinct_edges_rdd.collect()


all_nodes_trees = defaultdict(set)
edges_list = edges.collect()


distinct_indv_nodes = edges.flatMap(lambda x: [x[0], x[1]]).distinct().collect()




for two_nodes in edges_list:
    all_nodes_trees[two_nodes[0]].add(two_nodes[1])
    all_nodes_trees[two_nodes[1]].add(two_nodes[0])




    
def girvan_newman(all_nodes_trees, root_node):
    bfs_nodes = [root_node]
    dist = defaultdict(lambda: float('inf'))
    dist[root_node] = 0
    parents = defaultdict(set)
    n_paths = defaultdict(float)
    n_paths[root_node] = 1
    bottom_up_nodes = [root_node]

    for node in bfs_nodes:
        neighbors = all_nodes_trees[node]
        for neighbor in neighbors:
            if dist[neighbor] == float('inf'):
                dist[neighbor] = dist[node] + 1
                bfs_nodes.append(neighbor)
                bottom_up_nodes.append(neighbor)

            if dist[neighbor] == dist[node] + 1:
                n_paths[neighbor] += n_paths[node]
                parents[neighbor].add(node)

    betweenness = defaultdict(float)
    
    for edge in distinct_edges:
        betweenness[edge] = 0
        
    
        
    node_score = defaultdict(float)
        
    for node in bottom_up_nodes[::-1]:
        for parent in parents[node]:
            edge = (parent, node) if parent < node else (node, parent)
            credit = (n_paths[parent] / sum(n_paths[parent] for parent in parents[node])) * (1 + node_score[node])
            betweenness[edge] += credit
            node_score[parent] += credit

    return betweenness.items()

nodes_rdd = sc.parallelize(all_nodes_trees.keys())

betweenness_rdd = nodes_rdd.flatMap(lambda node: girvan_newman(all_nodes_trees, node))

betweenness_result = betweenness_rdd.reduceByKey(lambda a, b: a + b).filter(lambda x: x[1] != 0).mapValues(lambda x: x / 2).mapValues(lambda value: round(value, 5)).sortBy(lambda x: (-x[1], x[0])).map(lambda x: (tuple(sorted(x[0])), x[1])).collect()

with open(sys.argv[3], "w") as output_file:
    for x in betweenness_result:
        output_file.write(str(x[0])+ "," + str(x[1]) + "\n")


##################part2#############################################################################

def get_a_i_j (graph):
    a_values = defaultdict(int)
    for edge in graph:
        a_values[(edge[0],edge[1])]=1
        a_values[(edge[1],edge[0])]=1
        
    return a_values


def get_modularity(all_subgraphs, adjacencies, k_all, m):
    modularity = 0
    for subgraph in all_subgraphs:
        for i in subgraph:
            for j in subgraph:
                A_ij = adjacencies[(i,j)]
                k_i = k_all[i]
                k_j = k_all[j]
                modularity += A_ij - (k_i*k_j)/ (2*m)
    normalized_modularity = round(modularity / (2 * m), 7)
    return normalized_modularity

def find_communities(graph_rdd, all_nodes_trees, distinct_edges):
    
    m = len(distinct_edges)
    
    k_all = defaultdict(int)
    for edge in distinct_edges:
        k_all[edge[0]] +=1
        k_all[edge[1]] +=1
    
    adjacencies =  get_a_i_j(distinct_edges)
    
    best_modularity = -1
    best_communities = []
    
    nodes = []
    for edge in distinct_edges:
        for node in edge:
            if node not in nodes:
                nodes.append(node)
    
            

    
    while graph_rdd.count()>0:
        #print(len(graph_rdd.collect()))
        
        betweenness = defaultdict(float)
        for node in nodes:
            all_node_betweenness = girvan_newman(all_nodes_trees, node)
            for edge,i in all_node_betweenness:
                betweenness[edge] +=i
                
        for edge in betweenness:
            betweenness[edge] /=2
            
        highest_betweenness_now = -1
        edge_cut = None
        for key, value in betweenness.items():
            if value> highest_betweenness_now:
                highest_betweenness_now = value
                edge_cut = key
        
        #print("Edge to remove:", edge_cut)
        
        
        #print("Highest betweenness:", highest_betweenness_now)
            
        
        #graph_rdd = graph_rdd.filter(lambda edge: (edge[0], edge[1]) != edge_cut and (edge[1], edge[0]) != edge_cut)
        graph_rdd = graph_rdd.filter(lambda edge: (edge[0], edge[1]) != edge_cut)

        graph_rdd.cache()
        

        all_nodes_trees[edge_cut[0]].discard(edge_cut[1])
        all_nodes_trees[edge_cut[1]].discard(edge_cut[0])
        
        
        all_subgraphs = []
        
        seen = set()
        
        for node in nodes:
            if node not in seen:
                subgraph = set()
                
                new_nodes = [node]
                seen.add(node)
                
                while new_nodes:
                    current = new_nodes.pop()
                    subgraph.add(current)
                    
                    neighbors = list(all_nodes_trees[current])
                    
                    #neighbors = []
                    #for edge in all_nodes_trees[current]:
                    #    if edge[0] == current:
                    #        neighbors.append(edge[1])
                     #   else:
                     #       neighbors.append(edge[0])
                            
                    for neighbor in neighbors:
                        if neighbor not in seen:
                            seen.add(neighbor)
                            new_nodes.append(neighbor)
                            
                all_subgraphs.append(subgraph)
                
        modularity = get_modularity(all_subgraphs, adjacencies, k_all, m)
        
        if modularity > best_modularity:
            best_modularity = modularity
            best_communities = all_subgraphs
            
            
    return best_communities, best_modularity



best_communities, best_modularity = find_communities(distinct_edges_rdd, all_nodes_trees, distinct_edges)
sorted_best_communities = sorted(best_communities, key=lambda x: (len(x), sorted(x)[0]))

with open(sys.argv[4], "w") as output_file_2:
    for r in sorted_best_communities:
        output_file_2.write(', '.join("'" + user_id + "'" for user_id in sorted(r)) + "\n")

#print("Best communities:", best_communities)
#print("Max modularity:", best_modularity)
    
    
    
print(time.time()-start)

