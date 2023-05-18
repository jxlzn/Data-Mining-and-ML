import json
from pyspark import SparkContext
import os
import sys

#os.environ['PYSPARK_PYTHON'] = 'C:/Users/jxlzn/AppData/Local/Programs/Python/Python36/python.exe'
#os.environ['PYSPARK_DRIVER_PYTHON'] = 'C:/Users/jxlzn/AppData/Local/Programs/Python/Python36/python.exe'

sc = SparkContext('local[*]', 'task1')

textrdd = sc.textFile(sys.argv[1])

reviews = textrdd.map(json.loads)

output = {}
#a
total = reviews.count()
output["n_review"] = total

#b
filtered = reviews.filter(lambda x: '2018' in x["date"])
result = filtered.collect()
reviews2018 = len(result)
output["n_review_2018"]=reviews2018

#c
distinct = reviews.map(lambda x: x["user_id"]).distinct().count()
output["n_user"] = distinct

#d
top_users = reviews.map(lambda x: (x["user_id"], 1)).reduceByKey(lambda x, y: (x+y))
sorted_topusers = top_users.sortBy(lambda x: (-x[1], x[0]))
tuples = sorted_topusers.take(10)
final = []
for tuple in tuples:
    final.append(list(tuple))
output["top10_user"]= final

#e
distinctbiz = reviews.map(lambda x: x["business_id"]).distinct().count()
output["n_business"]=distinctbiz

#f
topbiz = reviews.map(lambda x: (x['business_id'], 1)).reduceByKey(lambda x, y: (x+y))
sorted_topbiz = topbiz.sortBy(lambda x: (-x[1], x[0]))
tuples2 = sorted_topbiz.take(10)
final2 = []
for tuple in tuples2:
    final2.append(list(tuple))
output["top10_business"]= final2

with open(sys.argv[2], "w") as file:
    json.dump(output, file, indent=4)







