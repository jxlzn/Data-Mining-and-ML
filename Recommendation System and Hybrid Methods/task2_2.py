import json
from pyspark import SparkContext, SparkConf
import os
import csv
from itertools import combinations
import sys
import time
import random
from operator import add
import operator
from math import sqrt
import json
import numpy as np
import xgboost 




#os.environ['PYSPARK_PYTHON'] = 'C:/Users/jxlzn/AppData/Local/Programs/Python/Python36/python.exe'
#os.environ['PYSPARK_DRIVER_PYTHON'] = 'C:/Users/jxlzn/AppData/Local/Programs/Python/Python36/python.exe'

conf = SparkConf().setMaster("local").setAppName("task2_3")
sc = SparkContext(conf = conf)


# Removed redundancies

training_file_path = sys.argv[1] + '/yelp_train.csv'
test_file_path = sys.argv[2]
output = sys.argv[3]

business_file_path = sys.argv[1] + '/business.json'
user_file_path = sys.argv[1] + '/user.json'

training_rdd = sc.textFile(training_file_path)
test_rdd = sc.textFile(test_file_path)
user_rdd = sc.textFile(user_file_path).map(lambda x: json.loads(x))
business_rdd = sc.textFile(business_file_path).map(lambda x: json.loads(x))

header_train = training_rdd.first()
header_test = test_rdd.first()

training_data = training_rdd.filter(lambda x: x != header_train).map(lambda x: x.split(","))
test_data_fromfile = test_rdd.filter(lambda x: x != header_test).map(lambda x: x.split(","))

train_new = training_data.map(lambda x: (x[1], x[0], x[2]))

userIds = train_new.map(lambda x: x[1]).distinct().sortBy(lambda x: x).zipWithIndex().collectAsMap()
businessIds = train_new.map(lambda x: x[0]).distinct().sortBy(lambda x: x).zipWithIndex().collectAsMap()

#users = set(userIds)
#businesses = set(businessIds)


#user_feature = user_rdd.filter(lambda x: x['user_id'] in users)
user_review_avg = user_rdd.map(lambda x: (x['user_id'], [x['average_stars']])).collectAsMap()


#business_feature = business_rdd.filter(lambda x: x['business_id'] in businesses)
business_review_stars = business_rdd.map(lambda x: (x['business_id'], [x['stars']])).collectAsMap()

user_review_avg_broadcast = sc.broadcast(user_review_avg)
business_review_stars_broadcast = sc.broadcast(business_review_stars)

def create_features(user_id, business_id, user_review_avg_broadcast, business_review_stars_broadcast):
    user_features = user_review_avg_broadcast.value.get(user_id, [0,0])
    business_features = business_review_stars_broadcast.value.get(business_id, [0,0])
    features = np.array([user_features, business_features]).flatten()
    
    return features

train_data = training_data.map(lambda x: (create_features(x[0], x[1], user_review_avg_broadcast, business_review_stars_broadcast), float(x[2]))).collect()
train_x = np.array([x[0] for x in train_data])
train_y = np.array([x[1] for x in train_data])


test_data = test_data_fromfile.map(lambda x: create_features(x[0], x[1], user_review_avg_broadcast, business_review_stars_broadcast))
test_x = np.array(test_data.collect())


xgb = xgboost.XGBRegressor()
xgb.fit(train_x, train_y)
predictions = xgb.predict(test_x)


test_pairs = test_data_fromfile.map(lambda x: (x[0], x[1])).collect()
results = [(pair[0], pair[1], prediction) for pair, prediction in zip(test_pairs, predictions)]




with open(output, mode='w', newline='') as file:
    file.write('user_id, business_id, prediction\n')
    for result in results:
        line = ",".join([str(x) for x in result]) + '\n'
        file.write(line)