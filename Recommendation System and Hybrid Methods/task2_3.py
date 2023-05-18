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

start = time.time()
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
test_data = test_rdd.filter(lambda x: x != header_test).map(lambda x: x.split(","))

train_new = training_data.map(lambda x: (x[1], x[0], x[2]))

userIds = train_new.map(lambda x: x[1]).distinct().sortBy(lambda x: x).zipWithIndex().collectAsMap()
businessIds = train_new.map(lambda x: x[0]).distinct().sortBy(lambda x: x).zipWithIndex().collectAsMap()

userIdsInvMap = {index: userId for userId, index in userIds.items()}
businessIdsInvMap = {index: businessId for businessId, index in businessIds.items()}


bus_user_dict = train_new.map(lambda x: (businessIds[x[0]], userIds[x[1]])).groupByKey().mapValues(set).collectAsMap()
user_bus_dict = train_new.map(lambda x: (userIds[x[1]], businessIds[x[0]])).groupByKey().mapValues(set).collectAsMap()

bus_rating_sum = train_new.map(lambda x: (businessIds[x[0]], float(x[2]))).reduceByKey(add)
bus_rating_count = train_new.map(lambda x: (businessIds[x[0]], 1)).countByKey()
bus_avg_rating = {k: v / bus_rating_count[k] for k, v in bus_rating_sum.collectAsMap().items()}

user_rating_sum = train_new.map(lambda x: (userIds[x[1]], float(x[2]))).reduceByKey(add)
user_rating_count = train_new.map(lambda x: (userIds[x[1]], 1)).countByKey()
user_avg_rating = {k: v / user_rating_count[k] for k, v in user_rating_sum.collectAsMap().items()}

business_user_rdd = train_new.map(lambda x: (businessIds[x[0]], (userIds[x[1]], float(x[2]))))
business_user_grouped = business_user_rdd.groupByKey()

business_user_rdd_dict = business_user_grouped.mapValues(dict).collectAsMap()

test_new = test_data.map(lambda x: (x[1], x[0]))
test_userIds = test_new.map(lambda x: x[1]).distinct().sortBy(lambda x: x).zipWithIndex().collectAsMap()
test_businessIds = test_new.map(lambda x: x[0]).distinct().sortBy(lambda x: x).zipWithIndex().collectAsMap()

test_users_dict = {index: user for user, index in test_userIds.items()}
test_business_dict = {index: business for business, index in test_businessIds.items()}




#user_feature = user_rdd.filter(lambda x: x['user_id'] in userIds)
user_review_avg = user_rdd.map(lambda x: (x['user_id'], [x['average_stars']])).collectAsMap()


#business_feature = business_rdd.filter(lambda x: x['business_id'] in businessIds)
business_review_stars = business_rdd.map(lambda x: (x['business_id'], [x['stars']])).collectAsMap()

#
pearson_dict = {}
k = 7

def calculate_pearson(item, b, corated_users):
    result1 = [float(business_user_rdd_dict[b][x]) for x in corated_users]
    result2 = [float(business_user_rdd_dict[item][x]) for x in corated_users]

    avg1, avg2 = sum(result1) / len(result1), sum(result2) / len(result2)

    numerator = sum((r1 - avg1) * (r2 - avg2) for r1, r2 in zip(result1, result2))
    denominator = sqrt(sum((r - avg1) ** 2 for r in result1)) * sqrt(sum((r - avg2) ** 2 for r in result2))

    return numerator / denominator if denominator != 0 else 0.0

def predict_ratings(b, u):
    if b not in bus_user_dict or u not in user_bus_dict:
        return user_avg_rating.get(u, 3.0)

    def get_weight(item):
        key = tuple(sorted((item, b)))
        w = pearson_dict.get(key)

        if w is None:
            corated_users = bus_user_dict[item] & bus_user_dict[b]
            w = (5.0 - abs(bus_avg_rating[b] - bus_avg_rating[item])) / 5 if len(corated_users) < k else calculate_pearson(item, b, corated_users)
            pearson_dict[key] = w

        return w, float(business_user_rdd_dict[item][u])

    pearson_list = [get_weight(item) for item in user_bus_dict[u]]
    top_pearson_corrs = sorted(pearson_list, key=lambda x: -x[0])[:600]
    numerator = sum(w * r for w, r in top_pearson_corrs)
    denominator = sum(abs(w) for w, r in top_pearson_corrs)

    return numerator / denominator if denominator != 0 else 3.0



def process_test_data(x):
    b, u = test_businessIds[x[0]], test_userIds[x[1]]
    rating = predict_ratings(b, u)
    user, business = test_users_dict[u], test_business_dict[b]
    return user, business, rating

result_item = test_new.map(process_test_data)


result_item = result_item.collect()





############# model based #################

def create_features(user_id, business_id, user_review_avg, business_review_stars):
    user_features = user_review_avg.get(user_id, [0])
    business_features = business_review_stars.get(business_id, [0])
    features = np.array([user_features, business_features]).flatten()
    return features

train_data = training_data.map(lambda x: (create_features(x[0], x[1], user_review_avg, business_review_stars), float(x[2]))).collect()
train_x = np.array([x[0] for x in train_data])
train_y = np.array([x[1] for x in train_data])
global_avg_rating = train_y.mean()

test_data_features = test_data.map(lambda x: create_features(x[0], x[1], user_review_avg, business_review_stars)).collect()
test_x = np.array(test_data_features)

xgb = xgboost.XGBRegressor()
xgb.fit(train_x, train_y)
predictions = xgb.predict(test_x)


test_pairs = test_data.map(lambda x: (x[0], x[1])).collect()
results_model = [(pair[0], pair[1], prediction) for pair, prediction in zip(test_pairs, predictions)]



def combine_results(result_item, results_model, weight_cf=0.3, weight_xgb=0.7):
    combined_results = []
    for i in range(len(result_item)):
        user, business, cf_rating = result_item[i]
        user_model, business_model, xgb_rating = results_model[i]
        combined_rating = (cf_rating * weight_cf) + (xgb_rating * weight_xgb)
        combined_results.append((user, business, combined_rating))
    return combined_results



final_predictions = combine_results(result_item, results_model)
#final_predictions_rdd = sc.parallelize(final_predictions)

#final_predictions_kv = final_predictions_rdd.map(lambda x: ((x[0], x[1]), x[2]))
#actual_ratings_kv = test_data.map(lambda x: ((x[0], x[1]), x[2]))

#joined_data = final_predictions_kv.join(actual_ratings_kv)
#squared_errors = joined_data.map(lambda x: (float(x[1][0]) - float(x[1][1]))**2)


#mse = squared_errors.mean()

#rmse = sqrt(mse)
#print("Root Mean Squared Error (RMSE):", rmse)

with open(output, mode='w', newline='') as file:
    file.write('user_id, business_id, prediction\n')
    for result in final_predictions:
        line = ",".join([str(x) for x in result]) + '\n'
        file.write(line)
