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



def to_int(user_id):
    return int(binascii.hexlify(user_id.encode('utf8')), 16)

def myhashs(user_id):
    userid = to_int(user_id)

    a1, b1, m1 = 131, 103, 69997
    a2, b2, m2, p2 = 137, 107, 69997, 701
    a3, b3, m3 = 149, 109, 69997
    hash1 = (a1 * userid + b1) % m1
    hash2 = ((a2 * userid + b2) % p2) % m2
    hash3 = (a3 * userid + b3) % m3

    return [hash1, hash2, hash3]

def bloom_filter(users, bits, users_seen):
    new_users = set()
    fp = 0

    for user in users:
        is_new = False
        for hash_value in myhashs(user):
            if bits[hash_value] == 0:
                is_new = True
                bits[hash_value] = 1
        if is_new:
            new_users.add(user)
        elif user not in users_seen:
            fp += 1

    return new_users, fp

num_of_asks = sys.argv[3]
stream_size = sys.argv[2]
file_name = sys.argv[1]
bits = [0] * 69997
users_seen = set()

bx = BlackBox()

with open(sys.argv[4], mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Time', 'FPR'])

    for ask_idx in range(int(num_of_asks)):
        stream = bx.ask(file_name, int(stream_size))
        new_users, fp = bloom_filter(stream, bits, users_seen)
        fpr = fp / len(new_users) if new_users else 0
        
        writer.writerow([ask_idx, fpr])
        users_seen.update(new_users)