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

from functools import reduce
from pyspark.sql.functions import col, lit, when
from graphframes import *


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


threshold = sys.argv[1]

def common_pairs(user_pair, threshold):
    user_i = user_pair[0]
    user_j = user_pair[1]

    common_businesses = user_i[1].intersection(user_j[1])

    if len(common_businesses) >= int(threshold):
        return [(user_i[0], user_j[0]), (user_j[0], user_i[0])]
    else:
        return []


def create_edges(user_business_list, threshold):

    user_pairs = user_business_list.cartesian(user_business_list).filter(lambda x: x[0][0] < x[1][0])


    edges = user_pairs.flatMap(lambda user_pair: common_pairs(user_pair, threshold))
    return edges


edges = create_edges(user_business_list, threshold)

edges_list = edges.flatMap(lambda x: [x[0], x[1]]).distinct().collect()


user_ids = set(edges_list)


vertices = user_business_list.map(lambda x: (x[0],)).filter(lambda x: x[0] in user_ids)


vertices_df = spark.createDataFrame(vertices, ["id"])
edges_df = spark.createDataFrame(edges, ["src", "dst"])

g = GraphFrame(vertices_df, edges_df)

result = g.labelPropagation(maxIter=5)


result_rdd = result.rdd

communities = result_rdd.map(lambda x: (x['label'], x['id'])).groupByKey().mapValues(lambda x: sorted(list(x)))

communities_sorted = communities.sortBy(lambda x: (len(x[1]), x[1]))

results = communities_sorted.map(lambda x: ', '.join("'" + user_id + "'" for user_id in x[1])).collect()


with open(sys.argv[3], "w") as output_file:
    for r in results:
        output_file.write(r + "\n")

print(time.time()-start)