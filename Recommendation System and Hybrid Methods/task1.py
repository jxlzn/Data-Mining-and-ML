import json
from pyspark import SparkContext, SparkConf
import os
import csv
from itertools import combinations
import sys
import time
import random
from operator import add


#os.environ['PYSPARK_PYTHON'] = 'C:/Users/jxlzn/AppData/Local/Programs/Python/Python36/python.exe'
#os.environ['PYSPARK_DRIVER_PYTHON'] = 'C:/Users/jxlzn/AppData/Local/Programs/Python/Python36/python.exe'

sc = SparkContext('local[*]', 'wordCount')
sc.setLogLevel('WARN')

start = time.time()

input_file_path = sys.argv[1]

datardd = sc.textFile(input_file_path)
header = datardd.first()
data = datardd.filter(lambda x: x != header).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1]))


print("0.5", time.time() - start)


userMap = data.map(lambda x: x[0]).distinct().zipWithIndex().collectAsMap()

businessMap = data.map(lambda x: x[1]).distinct().sortBy(lambda x: x).zipWithIndex().collectAsMap()


userIds = list(userMap.keys())
businessIds = list(businessMap.keys())

print("1", time.time() - start)

bu_main_rdd = data.map(lambda x: (businessMap[x[1]], [userMap[x[0]]])).reduceByKey(lambda x, y: x + y).sortBy(lambda x: x[0])

#
num_users = len(userIds)
num_business = len(businessIds)


h=100
b=50
r=2
k=10


print("2", time.time() - start)

def hashFunction(userList, hashv, num_users):
    return min([(((hashv[0] * usr) + hashv[1]) % num_users) for usr in userList])

hashvs = [(random.randint(1, num_users), random.randint(1, num_users)) for _ in range(h)]


sign_matrix = bu_main_rdd.map(lambda x: (x[0], [hashFunction(x[1], h, num_users) for h in hashvs]))


#def band_hash_function(signature, k):
#    return sum(signature) % k

band_signatures = sign_matrix.flatMap(
    lambda x: [
        ((band_index, tuple(x[1][band_index * r: (band_index + 1) * r])), [x[0]])
        for band_index in range(b)
    ]
)


grouped_band_signatures = band_signatures.reduceByKey(lambda x, y: x + y).filter(lambda x: len(x[1]) > 1)



print("3", time.time() - start)
candidate_pairs = grouped_band_signatures.flatMap(
    lambda x: [
     (tuple(sorted(comb)), 1) for comb in combinations(set(x[1]), 2)
    ]
).reduceByKey(add).map(lambda x: x[0])




bu_list = bu_main_rdd.collect()


print("4", time.time() - start)



similar_pairs = candidate_pairs.map(lambda pair: (
        businessIds[pair[0]],
        businessIds[pair[1]],
        len(set(bu_list[pair[0]][1]) & set(bu_list[pair[1]][1])) / len(set(bu_list[pair[0]][1]) | set(bu_list[pair[1]][1])))).filter(lambda x: x[2] >= 0.5)

print("5", time.time() - start)




sorted_pairs = similar_pairs.sortBy(lambda x: (x[0], x[1]))




print("6", time.time() - start)
with open(sys.argv[2], 'w') as f:
    f.write("business_id_1, business_id_2, similarity\n")
    for pair in sorted_pairs.collect():
        f.write(f"{pair[0]},{pair[1]},{pair[2]}\n")

end = time.time()
print("Duration: ", end - start)


