from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.linalg import Vectors
from pyspark.sql import Row
from sklearn.neighbors import NearestNeighbors
import numpy as np
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
import pandas as pd
from pyspark.rdd import RDD
import os

#os.path.abspath('/media/vishwanath/Vishwanath/IIITB/Sem_2/PE/Data/Training_data.csv')
sc = SparkContext("local", "test_script")
sqlContext = SQLContext(sc)
data = sc.textFile("Training_Data.csv")
pddata = pd.read_csv('Training_Data.csv')

#Remove Header
header = data.first()
data = data.filter(lambda line: line != header)

classes = pd.Series(pddata['class'])

def transformToLabelledPoint(inputStr):
    attlist = inputStr.split(",")
    data = (float(attlist[2]) - float(attlist[4]),
         float(attlist[5]) - float(attlist[4]),
         float(attlist[4]) - float(attlist[3]),
         float(attlist[3]) - float(attlist[6]))
    return data

def tranformToClass(inputStr):
    attlist = inputStr.split(",")
    classes.append(attlist[1])
    return classes

autolp = data.map(transformToLabelledPoint)
autolpCollect = autolp.collect()
##autoDF = sqlContext.createDataFrame(autolp, ["g-r","u-r","r-i","i-z"])
vectorsCollected = np.vstack(tuple(autolpCollect))

#classes = data.map(tranformToClass)
##classDF = sqlContext.createDataFrame(classes, ["class"])
##classDF.select("class").show(10)

##(trainingData, testData) = autoDF.randomSplit([0.8,0.2])
##df = trainingData.toPandas()
#print df
##df2 = testData.toPandas()
nbrs = NearestNeighbors(n_neighbors=7, algorithm='auto').fit(vectorsCollected)
bc_knnobj = sc.broadcast(nbrs)

##distances, indices = nbrs.kneighbors(vectorsCollected)
results = autolp.map(lambda x: bc_knnobj.value.kneighbors(x))
print results.take(2)

def accessList(x):
    (x[0].item(0,0))
    

resultList = results.map(accessList)
print resultList.take(2)

##predicted_class = []
##for i in range(indices.size/7):
##    quasar_count = 0
##    non_quasar_count = 0
##    for j in range(indices[i].size):
##        if(classes[indices[i,j]] == 'QSO'):
##            quasar_count += (1/(distances[i,j]*distances[i,j]))
##        else:
##            non_quasar_count += (1/(distances[i,j]*distances[i,j]))
##    if(quasar_count > non_quasar_count):
##        predicted_class.append('QSO')
##    else:
##        predicted_class.append('NQSO')
##print predicted_class
##autoDF.printSchema()
