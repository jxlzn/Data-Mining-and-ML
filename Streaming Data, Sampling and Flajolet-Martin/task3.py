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

random.seed(553)

num_of_asks = int(sys.argv[3])
stream_size = int(sys.argv[2])
file_name = sys.argv[1]

bx = BlackBox()
reservoir = []
n = 0


with open(sys.argv[4], mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['seqnum','0_id','20_id','40_id','60_id','80_id'])
    
    for i in range(num_of_asks):
        stream_users = bx.ask(file_name, stream_size)

        for new_user in stream_users:
            n += 1
            if len(reservoir) < 100:
                reservoir.append(new_user)
            else:
                if random.random() < 100/n:
                    replace = random.randint(0, 99)
                    reservoir[replace] = new_user

        idx = [0, 20, 40, 60, 80]
        output_users = [reservoir[i] for i in idx]
        writer.writerow([n] + output_users)

