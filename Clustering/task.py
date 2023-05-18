from pyspark import SparkConf, SparkContext
from sklearn.cluster import KMeans
import numpy as np
import os
from sklearn.cluster import DBSCAN, MeanShift, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import sys
import time
import itertools
import random
from collections import defaultdict

#os.environ['PYSPARK_PYTHON'] = 'C:/Users/jxlzn/AppData/Local/Programs/Python/Python36/python.exe'
#os.environ['PYSPARK_DRIVER_PYTHON'] = 'C:/Users/jxlzn/AppData/Local/Programs/Python/Python36/python.exe'

def parse_line(line):
    items = [float(x) for x in line.split(',')]
    return items[0], items[1], np.array(items[2:])

with open(sys.argv[1], 'r') as f:
    data = [parse_line(line.strip()) for line in f.readlines()]

# data_20 = random.sample(data, int(0.2 * len(data)))
# data_80 = [item for item in data if item not in data_20]

chosen_indices = set(np.random.choice(len(data), int(0.2 * len(data)), replace=False))
data_20 = [data[i] for i in chosen_indices]
#print(data_20[:5])
data_80 = [data[i] for i in range(len(data)) if i not in chosen_indices]
#
#
#
data_20_features = np.array([x[2] for x in data_20])
#print(data_20_features[:5])
n_cluster = int(sys.argv[2])
#
kmeans_50 = KMeans(n_clusters=n_cluster*5, random_state=42).fit(data_20_features)
labels = kmeans_50.labels_
#
#
data_20_with_labels = [(data_20[i][0], data_20[i][1], data_20[i][2], labels[i]) for i in range(len(data_20))]

# #
cluster_counts = {label: list(labels).count(label) for label in set(labels)}

single_point_clusters = {cluster for cluster, count in cluster_counts.items() if count == 1}
# #single_point_clusters = {cluster for cluster, count in cluster_counts.items() if count == 1 or cluster == -1}
retained_set_step3 = [x for x in data_20_with_labels if x[3] in single_point_clusters]
# print(len(retained_set))
remaining_data = [x for x in data_20_with_labels if x[3] not in single_point_clusters]
# print(len(remaining_data))
# #
remaining_features = np.array([x[2] for x in remaining_data])
#
kmeans_10 = KMeans(n_clusters=n_cluster, random_state=42).fit(remaining_features)

labels_10 = kmeans_10.labels_
#
remaining_data_with_labels_10 = [(remaining_data[i][0], remaining_data[i][1], remaining_data[i][2], labels_10[i]) for i in range(len(remaining_data))]



discard_points_dict = {i: [] for i in range(10)}
discard_stats_dict = {}

# Add points to the corresponding cluster in the dictionary
for point in remaining_data_with_labels_10:
    label = point[3]
    discard_points_dict[label].append(point)



#print(discard_points_dict)


for cluster_label, points in discard_points_dict.items():
    features_only = points[2]
    points_np = np.array(features_only)
    num_points = len(points_np)
    sum_coords = np.sum(points_np, axis=0)
    sum_squares_coords = np.sum(np.square(points_np), axis=0)
    discard_stats_dict[cluster_label] = (num_points, sum_coords, sum_squares_coords)



num_discard_points = sum(len(points) for points in discard_points_dict.values())

#print(num_discard_points)
#
retained_features = np.array([x[2] for x in retained_set_step3])


compressed_points_dict = {}
compressed_stats_dict = {}

if len(retained_set_step3) >n_cluster*5:
    kmeans_retained = KMeans(n_clusters=n_cluster*5, random_state=42).fit(retained_features)

    labels_retained = kmeans_retained.labels_
    #
    retained_set_with_labels = [(retained_set_step3[i][0], retained_set_step3[i][1], retained_set_step3[i][2], retained_set_step3[i][3], labels_retained[i]) for i in range(len(retained_set_step3))]



    cluster_counts_10 = {label: list(labels_retained).count(label) for label in set(labels_retained)}
    single_point_clusters_10 = {cluster for cluster, count in cluster_counts_10.items() if count == 1}

    updated_retained_set = [x for x in retained_set_with_labels if x[4] in single_point_clusters_10]

    compressed_set = [x for x in retained_set_with_labels if x[4] not in single_point_clusters_10]



    
    for point in compressed_set:
        label = point[4]
        compressed_points_dict[label].append(point)

    
    for cluster_label, points in compressed_points_dict.items():
        features_only = [x[2] for x in points]
        points_np = np.array(features_only)
        num_points = len(points_np)
        sum_coords = np.sum(points_np, axis=0)
        sum_squares_coords = np.sum(np.square(points_np), axis=0)
        compressed_stats_dict[cluster_label] = (num_points, sum_coords, sum_squares_coords)

else:
    updated_retained_set = retained_set_step3


#print("updated_retained_set", updated_retained_set)
retained_points_dict = {point[0]: point for point in updated_retained_set}

num_compression_clusters_20 = 0
num_compression_points_20 = 0

for cluster_label, points in compressed_points_dict.items():
    if len(points) > 0:
        num_compression_clusters_20 += 1
        num_compression_points_20 += len(points)

#print("Number of compression sets:", num_compression_clusters_20)
#print("Total number of points in compression sets:", num_compression_points_20)


num_retained_points_20 = len(updated_retained_set)
#print(num_retained_points_20)


#print(len(discard_points_dict))

num_discard_points_20 = sum(len(points) for points in discard_points_dict.values())
# #
# #
# # #######################step7#######################################################
# #
def mahalanobis_distance_direct(x, num_points, sum_coords, sum_squares_coords):

    features = x  # Extract features from the data point
    mean = sum_coords / num_points
    variances = (sum_squares_coords / num_points) - np.square(mean)
    std_deviations = np.sqrt(variances)
    normalized_diff = (features - mean) / std_deviations
    return np.sqrt(np.sum(np.square(normalized_diff)))

def assign_point(point, discard_points_dict, discard_stats_dict, compressed_points_dict, compressed_stats_dict):
    global num_features
    num_features = len(point[2])

    
    discard_distances = [mahalanobis_distance_direct(point[2], *discard_stats_dict[ds]) for ds in discard_stats_dict]
    min_discard_distance, discard_set_index = min((val, idx) for (idx, val) in enumerate(discard_distances))

    if min_discard_distance < 2 * np.sqrt(num_features):
        point = list(point)
        point.append(discard_set_index)
        point = tuple(point)
        discard_points_dict[discard_set_index].append(point)

        # Update discard_stats_dict
        num_points, sum_coords, sum_squares_coords = discard_stats_dict[discard_set_index]
        num_points += 1
        sum_coords += point[2]
        sum_squares_coords += np.square(point[2])
        discard_stats_dict[discard_set_index] = (num_points, sum_coords, sum_squares_coords)
        return

   
    compressed_distances = [mahalanobis_distance_direct(point[2], *compressed_stats_dict[cs]) for cs in compressed_stats_dict]
    min_compressed_distance, compressed_set_index = min((val, idx) for (idx, val) in enumerate(compressed_distances))

    if min_compressed_distance < 2 * np.sqrt(num_features):
        compressed_points_dict[compressed_set_index].append(point)

        # Update compressed_stats_dict
        num_points, sum_coords, sum_squares_coords = compressed_stats_dict[compressed_set_index]
        num_points += 1
        sum_coords += point[2]
        sum_squares_coords += np.square(point[2])
        compressed_stats_dict[compressed_set_index] = (num_points, sum_coords, sum_squares_coords)
        return

    
    retained_points_dict[point[0]] = point
#
# #
# #


def merge_cs_clusters(compressed_points_dict, compressed_stats_dict):

    for cid1, values1 in list(compressed_stats_dict.items()):
        for cid2, values2 in list(compressed_stats_dict.items()):
            if cid1 < cid2:
                mahalanobis_dis = np.sqrt(np.sum(np.square(np.divide(np.subtract(values1[1], values2[1]), np.add(np.sqrt(values1[2]), np.sqrt(values2[2])))), axis=0))
                if mahalanobis_dis >= 2 * np.sqrt(num_features):
                   continue
                else:
                    # Merge two CS clusters
                    n = values1[0] + values2[0]
                    SUM = np.add(values1[1], values2[1])
                    SUMSQ = np.add(values1[2], values2[2])
                    compressed_stats_dict[cid1] = (n, SUM, SUMSQ)
                    compressed_points_dict[cid1] += compressed_points_dict.pop(cid2, [])
                    compressed_stats_dict.pop(cid2)

    return compressed_points_dict, compressed_stats_dict


def merge_cs_to_ds(discard_points_dict, discard_stats_dict, compressed_points_dict, compressed_stats_dict):
    global num_features
    threshold = 2 * np.sqrt(num_features)

    
    for cs_label, cs_points in list(compressed_points_dict.items()):
        cs_stats = compressed_stats_dict[cs_label]

       
        discard_distances = [mahalanobis_distance_direct(cs_stats[1] / cs_stats[0], *discard_stats_dict[ds]) for ds in discard_stats_dict]
        min_discard_distance, discard_set_index = min((val, idx) for (idx, val) in enumerate(discard_distances))

        
        if min_discard_distance < threshold:
            # Merge points
            discard_points_dict[discard_set_index].extend(cs_points)

            # Merge stats
            discard_stats_dict[discard_set_index] = (
                discard_stats_dict[discard_set_index][0] + cs_stats[0],
                discard_stats_dict[discard_set_index][1] + cs_stats[1],
                discard_stats_dict[discard_set_index][2] + cs_stats[2]
            )

            
            del compressed_points_dict[cs_label]
            del compressed_stats_dict[cs_label]

    return discard_points_dict, discard_stats_dict, compressed_points_dict, compressed_stats_dict

#
#
#     index_map = {i: i for i in range(len(cluster_stats))}
#     if min_distance < threshold:
#
#         index_map[j] = i
#         cluster_stats.pop(j)
#
#     return cluster_stats, index_map
#
#
#
#
#
def process_data_chunk(data_chunk, discard_points_dict, discard_stats_dict, compressed_points_dict, compressed_stats_dict, retained_points_dict):
    for point in data_chunk:
        assign_point(point, discard_points_dict, discard_stats_dict, compressed_points_dict, compressed_stats_dict)
#         if retained_point is not None:
#             retained_set_new.append(retained_point)




    if len(retained_set_step3) >n_cluster*5:
        retained_features_new = [point[2] for point in retained_points_dict.values()]
        print(retained_features_new)
        kmeans_retained_new = KMeans(n_clusters=n_cluster*5).fit(retained_features_new)
        retained_labels_new = kmeans_retained_new.labels_
    #
        retained_set_with_labels_new = [(point[0], point[1], point[2], label) for point, label in zip(retained_points_dict.values(), retained_labels_new)]


        


        cluster_counts_step11 = {label: list(retained_labels_new).count(label) for label in set(retained_labels_new)}
        single_point_clusters_step11 = {cluster for cluster, count in cluster_counts_step11.items() if count == 1}

        retained_set_step11 = [x for x in retained_set_with_labels_new if x[4] in single_point_clusters_step11]

        compressed_set_step11 = [x for x in retained_set_with_labels_new if x[4] not in single_point_clusters_step11]




        compressed_points_dict_step11 = defaultdict(list)
        for point in compressed_set_step11:
            label = point[4]
            compressed_points_dict_step11[label].append(point)

        
        compressed_points_dict_step11 = {}
        for cluster_label, points in compressed_points_dict_step11.items():
            features_only = [x[2] for x in points]
            points_np = np.array(features_only)
            num_points = len(points_np)
            sum_coords = np.sum(points_np, axis=0)
            sum_squares_coords = np.sum(np.square(points_np), axis=0)
            compressed_points_dict_step11[cluster_label] = (num_points, sum_coords, sum_squares_coords)

        
        compressed_points_dict.update(compressed_points_dict_step11)


        compressed_stats_dict.update(compressed_stats_dict_step11)


        compressed_points_dict, compressed_stats_dict = merge_cs_clusters(compressed_points_dict, compressed_stats_dict)





#
num_partitions = 4

# Shuffle the data_80 list
random.shuffle(data_80)

partition_size = len(data_80) // num_partitions
data_chunks = [data_80[i * partition_size:(i + 1) * partition_size] for i in range(num_partitions)]
#
# discard_point_labels = {}
#
#
#
with open(sys.argv[3], "w") as f:
    f.write("The intermediate results:\n")

    f.write("Round 1: ")
    f.write(f"{num_discard_points_20}, {num_compression_clusters_20}, {num_compression_points_20}, {num_retained_points_20}\n")
#


for i, data_chunk in enumerate(data_chunks, start=2):

    process_data_chunk(data_chunk, discard_points_dict, discard_stats_dict, compressed_points_dict, compressed_stats_dict, retained_points_dict)
    
    if i == 5:
        discard_points_dict, discard_stats_dict, compressed_points_dict, compressed_stats_dict = merge_cs_to_ds(discard_points_dict, discard_stats_dict, compressed_points_dict, compressed_stats_dict)
    
    
    
    num_discard_points = sum(len(points) for points in discard_points_dict.values())

    num_compressed_points = sum(len(points) for points in compressed_points_dict.values())



    #print(discard_points_dict)


    




#         #chunk_labels, *stats = process_data_chunk(data_chunk, discard_stats, compressed_set_stats)
    with open(sys.argv[3], "a") as f:
        f.write(f"Round {i}: ")
        f.write(f"{num_discard_points}, {len(compressed_points_dict)}, {num_compressed_points}, {len(retained_points_dict)}\n")
# #
# #



with open(sys.argv[3], "a") as f:
    f.write("\n")
    f.write("The clustering results:\n")




    #print(discard_points_dict)
    cluster_assignments = {}

    for key, value in discard_points_dict.items():
        for point in value:


            cluster_assignments[int(point[0])] = key


    for key, value in compressed_points_dict.items():
        for point in value:
            cluster_assignments[int(point[0])] = -1

    for rid in retained_points_dict.keys():
        cluster_assignments[int(rid)]= -1



    for point, cluster in sorted(cluster_assignments.items(), key=lambda x: x[0]):
        f.write(f"{point},{cluster}\n")





