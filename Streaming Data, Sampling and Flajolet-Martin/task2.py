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
from functools import reduce
from blackbox import BlackBox
import binascii
import random
import statistics
from math import log2


def to_int(user_id):
    return int(binascii.hexlify(user_id.encode('utf8')), 16)

def get_hash_funcs(num_functions):
    return [(random.randint(1, 100), random.randint(1, 100), 69997) for _ in range(num_functions)]

def myhashs(s):
    return [(a * to_int(s) + b) % m for a, b, m in hash_functions]


def count_zeros(value):
    count = 0
    while value and value & 1 == 0:
        count += 1
        value >>= 1
    return count

def flajolet_martin(users, hash_functions):
    R = [0] * len(hash_functions)
    for user in users:
        for i, hashed_value in enumerate(myhashs(user)):
            R[i] = max(R[i], count_zeros(hashed_value))
    return R

def count_distinct(R, num_groups):
    
    group_size = len(R) // num_groups

 
    averages = []

    
    for i in range(num_groups):
        
        start = i * group_size
        end = (i + 1) * group_size

        group = R[start:end]

        group_sum = sum(group)

        group_avg = group_sum / group_size

        estimate = 2 ** group_avg

        averages.append(estimate)

    return statistics.median(averages)

num_of_asks = sys.argv[3]
stream_size = sys.argv[2]
file_name = sys.argv[1]
num_hash_functions = 32
num_groups = 4

hash_functions = get_hash_funcs(num_hash_functions)
bx = BlackBox()

with open(sys.argv[4], mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Time','Ground Truth','Estimation'])

    for ask_idx in range(int(num_of_asks)):
        stream = bx.ask(file_name, int(stream_size))
        R = flajolet_martin(stream, hash_functions)
        estimated = int(count_distinct(R, num_groups))
        ground_truth = len(set(stream))
        
        writer.writerow([ask_idx, ground_truth, estimated])