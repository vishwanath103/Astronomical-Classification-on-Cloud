from pyspark import SparkContext
from pyspark.sql import SQLContext
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os

#os.chdir("/home/bailorevishwanath02")
sc = SparkContext("local", "test_script")
sqlContext = SQLContext(sc)

trainingFile = "gs://vishu/TrainingData/Training_Data.csv"
data = sc.textFile(trainingFile)
#pddata = pd.read_csv(trainingFile)

#Remove Header
header = data.first()
data = data.filter(lambda line: line != header)


def transformToClass(inputStr):
    attlist = inputStr.split(",")
    lp = (float(attlist[0]),attlist[1])
    return lp

classlp = data.map(transformToClass)
classDF = sqlContext.createDataFrame(classlp, ["id","class"])
dfclass = classDF.toPandas()

classes = pd.Series(dfclass['class'])
#print classes

def transformToLabelledPoint(inputStr):
    attlist = inputStr.split(",")
    lp = (float(attlist[2]) - float(attlist[4]),
         float(attlist[5]) - float(attlist[4]),
         float(attlist[4]) - float(attlist[3]),
         float(attlist[3]) - float(attlist[6]))
    return lp

autolp = data.map(transformToLabelledPoint)
autolpCollect = autolp.collect()
vectorsCollected = np.vstack(tuple(autolpCollect))

classes = data.map(tranformToClass)
classDF = sqlContext.createDataFrame(classes, ["id","class"])
dfclass = classDF.toPandas()
classPD = pd.Series(dfclass['class'])

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(vectorsCollected, classPD)

testingFile = "gs://vishu/TestingData/"
testData = sc.textFile(testingFile)

bc_knnobj = sc.broadcast(knn)

def transformTestData(inputStr):
    attlist = inputStr.split(",")
    data = (float(attlist[4]) - float(attlist[6]),
         float(attlist[7]) - float(attlist[6]),
         float(attlist[6]) - float(attlist[5]),
         float(attlist[5]) - float(attlist[8]))
    return data

testlp = testData.map(transformTestData)
results = testlp.map(lambda x: bc_knnobj.value.predict(x))
##print distances


