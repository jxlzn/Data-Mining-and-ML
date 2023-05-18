########### Task 2

import json
from pyspark import SparkContext
import os
import time
import sys



#os.environ['PYSPARK_PYTHON'] = 'C:/Users/jxlzn/AppData/Local/Programs/Python/Python36/python.exe'
#os.environ['PYSPARK_DRIVER_PYTHON'] = 'C:/Users/jxlzn/AppData/Local/Programs/Python/Python36/python.exe'

sc = SparkContext('local[*]', 'task2')

input_file_path = sys.argv[1]
textrdd = sc.textFile(input_file_path)

reviews = textrdd.map(json.loads)

start_time = time.time()
topbiz = reviews.map(lambda x: (x['business_id'], 1)).reduceByKey(lambda x, y: (x+y)).sortBy(lambda x: (-x[1], x[0]))


output = {}
output['default']= {}
output['customized'] = {}

output['default']['n_partition'] = topbiz.getNumPartitions()
output['default']['n_items'] = topbiz.glom().map(len).collect()
output['default']['exe_time']= time.time() - start_time



# print(topbiz.take(5))
# print("Number of partitions: {}".format(topbiz.getNumPartitions()))
# print("Partitioner: {}".format(topbiz.partitioner))
# print("Partitions structure: {}".format(topbiz.glom().collect()))

# print(f"Execution time: {time.time() - start_time}")


start_time2 = time.time()

topbiz2 = reviews.map(lambda x: (x['business_id'], 1)).partitionBy(numPartitions=int(sys.argv[3]), partitionFunc=lambda key: hash(key) % 100).reduceByKey(lambda x, y: (x+y)).sortBy(lambda x: (-x[1], x[0]))
print("Number of partitions: {}".format(topbiz2.getNumPartitions()))
# print("Partitioner: {}".format(topbiz2.partitioner))
# print("Partitions structure: {}".format(topbiz2.glom().collect()))

output['customized']['n_partition'] = topbiz2.getNumPartitions()
output['customized']['n_items'] = topbiz2.glom().map(len).collect()
output['customized']['exe_time']= time.time() - start_time2

with open(sys.argv[2], "w") as f:
    json.dump(output, f)
# print(f"Execution time: {time.time() - start_time2}")