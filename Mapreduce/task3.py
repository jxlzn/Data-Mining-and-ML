import json
from pyspark import SparkContext
import os
import time
import sys




#os.environ['PYSPARK_PYTHON'] = 'C:/Users/jxlzn/AppData/Local/Programs/Python/Python36/python.exe'
#os.environ['PYSPARK_DRIVER_PYTHON'] = 'C:/Users/jxlzn/AppData/Local/Programs/Python/Python36/python.exe'

sc = SparkContext('local[*]', 'task3')

input_file_path = sys.argv[1]
text = sc.textFile(input_file_path)

reviews = text.map(json.loads).map(lambda x: (x["business_id"], x["stars"]))

businessRDD = sc.textFile(sys.argv[2])

business = businessRDD.map(json.loads).map(lambda x: (x["business_id"], x["city"]))

joined = reviews.join(business).map(lambda x: (x[1][1], x[1][0]))

grouped = joined.groupByKey().map(lambda x: (x[0], sum(x[1])/len(x[1])))

sorted = grouped.sortBy(lambda x: (-x[1], x[0]))

answer = sorted.collect()

first_line = "city, stars\n"
with open(sys.argv[3], "w") as file:
    file.write(first_line)
    for i,o in answer:
        line = i + ", " + str(o) + "\n"
        file.write(line)


start_time = time.time()
sorted1 = grouped.sortBy(lambda x: (-x[1], x[0]))
print(sorted1.take(10))
spark_time = time.time() - start_time

start_time = time.time()
sorted2 = list(grouped.collect())
sorted2.sort(key=lambda x: (-x[1], x[0]))
print(sorted2[:10])
python_time = time.time() - start_time

all_times = {"m1": python_time, "m2": spark_time, "reason": "The spark time took a bit longer than the python time, possibly because for a smaller dataset, spark sorts via different nodes that are working in parallel, but python is able to sort using a single node operation, thus reducing the need for longer communication times in the spark sorting algorithms."}
with open(sys.argv[4], "w") as f:
    json.dump(all_times, f)
    
    
    
