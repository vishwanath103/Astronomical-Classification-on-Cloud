from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.linalg import Vectors
from pyspark.sql import Row
from sklearn.neighbors import NearestNeighbors
import numpy as np
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
import pandas as pd
import os

sc = SparkContext("local", "test_script")
sqlContext = SQLContext(sc)
data = sc.textFile("sample.csv")

#Remove Header
##header = data.first()
##data = data.filter(lambda line: line != header)

def transformToLabelledPoint(inputStr):
    attlist = inputStr.split(",")
    vector = [float(attlist[2]) - float(attlist[4]),
                           float(attlist[5]) - float(attlist[4]),
                           float(attlist[4]) - float(attlist[3]),
                           float(attlist[3]) - float(attlist[6])]
    return vector

vectors = data.map(transformToLabelledPoint)
vectors.collect()
##df = pd.DataFrame(d)
##
##(trainingData, testData) = df.randomSplit([0.8,0.2])
nbrs = NearestNeighbors(n_neighbors=7, algorithm='auto').fit(vectors)
##
##
##autoDF.printSchema()
