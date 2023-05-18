import json
from pyspark import SparkContext, SparkConf
import os
import csv
from itertools import combinations
import sys
import time


#os.environ['PYSPARK_PYTHON'] = 'C:/Users/jxlzn/AppData/Local/Programs/Python/Python36/python.exe'
#os.environ['PYSPARK_DRIVER_PYTHON'] = 'C:/Users/jxlzn/AppData/Local/Programs/Python/Python36/python.exe'

with open(sys.argv[3], 'r') as file:
    reader = csv.reader(file)
    next(reader)
    rows = []
    for r in reader:

        date = r[0].replace('20', '')


        newcol = date + '-' + r[1].strip()

        id = str(int(r[5]))

        rows.append([newcol, id])




with open('Customer_product.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['DATE-CUSTOMER_ID', 'PRODUCT_ID'])
    for row in rows:
        writer.writerow(row)

#conf = SparkConf().setAppName("MyApp").set("spark.executor.memory", "4g")
#sc = SparkContext(conf=conf)
# sc = SparkContext('local[*]', 'wordCount')
sc = SparkContext('local[*]', 'task2')
sc.setLogLevel('WARN')




input_file_path = 'Customer_product.csv'


rawrdd = sc.textFile(input_file_path, 8)
header = rawrdd.first()


start = time.time()

threshold = int(sys.argv[1])
support = int(sys.argv[2])



collected_all = rawrdd.filter(lambda x: x!= header).map(lambda x: x.split(',')).map(lambda x: (x[0], [x[1]])).reduceByKey(lambda x, y: x+y).filter(lambda x: len(x[1])> threshold)


baskets = collected_all.map(lambda x: x[1])


baskets_all = baskets.collect()


candidate = []
frequent_all=[]


lengths = baskets.map(lambda x: len(x))

max_size = lengths.max()

unique_items = baskets.flatMap(lambda x: x).distinct()



unique_items_list = unique_items.filter(lambda x: not x.startswith('1')).sortBy(lambda x: x).collect()
unique_items_list = ["('" + item + "')" for item in unique_items_list]

candidate.append(unique_items_list)

frequent_size1_pre = unique_items.map(lambda x: (x, sum(1 for y in baskets_all if x in y))).filter(lambda x: x[1] >= int(support)).sortBy(lambda x: x)

candidate.append(frequent_size1_pre.collect())

frequent_size1= frequent_size1_pre.map(lambda x: set([x[0]]))


frequent_size1 = frequent_size1.map(lambda x: tuple(sorted(x))).distinct().sortBy(lambda x: x)

frequent_size1_list = frequent_size1.collect()
frequent_size1_list = ["('" + str(tuple(item)[0]) + "')" for item in frequent_size1_list]




frequent_all.append(frequent_size1_list)

items = frequent_size1.flatMap(lambda x: x).collect()


candidate_size2 = sc.parallelize(combinations(items, 2)).map(lambda x: tuple(sorted(x)))

candidate.append(candidate_size2.collect())


frequent_size2 = candidate_size2.map(lambda x: (x, sum(1 for i in baskets_all if set(x).issubset(set(i))))).filter(lambda x: x[1] >= support).map(lambda x: tuple(x[0]))

frequent_size2 = frequent_size2.map(lambda x: tuple(sorted(x))).distinct().sortBy(lambda x: x)


frequent_all.append(frequent_size2.collect())

k = 3
frequent_sizek_minus1 = frequent_size2.map(lambda x: set(x))
frequent_sizek_list = frequent_sizek_minus1.collect()


while frequent_sizek_minus1.count() > 0 and k <= max_size:
    candidate_sizek = frequent_sizek_minus1.flatMap(lambda x: [(y.union(x)) for y in frequent_sizek_list if len(y.union(x)) == k])
    candidate.append(candidate_sizek.map(lambda x: tuple(sorted(x))).distinct().collect())


    def count_frequent(x):
        return (x, sum(1 for _ in baskets_all if x.issubset(set(_))))

    frequent_sizek = candidate_sizek.map(count_frequent).filter(lambda x: x[1] >= support).map(lambda x: tuple(x[0]))

    frequent_sizek = frequent_sizek.map(lambda x: tuple(sorted(x))).distinct().sortBy(lambda x: x)


#     print(frequent_sizek.collect())
    frequent_all.append(frequent_sizek.collect())
    frequent_sizek_minus1 = frequent_sizek.map(lambda x: set(x))
    frequent_sizek_list = frequent_sizek_minus1.collect()
    k += 1
# #
# # frequent_itemsets_all = frequent_itemsets_size1.union(frequent_itemsets_size2).union(frequent_itemsets_sizek)


with open(sys.argv[4], 'w') as f:
    f.write("Candidates:\n")
    previous_len = 0
    for cand in candidate:
        for item_set in cand:
            f.write(str(item_set) + ", ")

        if len(cand)> previous_len:
            f.seek(f.tell() - 2)
            f.write("\n\n")


    f.write("Frequent Itemsets:\n")

    previous_len = 0
    for freq_set in frequent_all:
        for item_set in freq_set:
            f.write(str(item_set) + ", ")

        if len(freq_set)> previous_len:
            f.seek(f.tell() - 2)
            f.write("\n\n")



end = time.time()
print("Duration: ", end - start)