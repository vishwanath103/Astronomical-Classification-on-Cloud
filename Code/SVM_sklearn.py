from pyspark import SparkContext
from pyspark.sql import SQLContext
import pandas as pd
from sklearn import svm
import numpy as np

sc = SparkContext("local", "test_script")
sqlContext = SQLContext(sc)
trainingFile = "gs://vishu/TrainingData/Training_Data.csv"
data = sc.textFile(trainingFile)

#Remove Header
header = data.first()
trianData = data.filter(lambda line: line != header)

def transformToLabelledPoint(inputStr):
    attlist = inputStr.split(",")
    data = (float(attlist[2]) - float(attlist[4]),
         float(attlist[5]) - float(attlist[4]),
         float(attlist[4]) - float(attlist[3]),
         float(attlist[3]) - float(attlist[6]))
    return data

def tranformToClass(inputStr):
    attlist = inputStr.split(",")
    classes = (float(attlist[0]), attlist[1])
    return classes

datalp = data.map(transformToLabelledPoint)
datalpCollect = datalp.collect()
vectorsCollected = np.vstack(tuple(datalpCollect))

classes = data.map(tranformToClass)
classDF = sqlContext.createDataFrame(classes, ["id","class"])
dfclass = classDF.toPandas()
classPD = pd.Series(dfclass['class'])


rbf_svm = svm.SVC(kernel='rbf')

print "before fit"
rbf_svm.fit(vectorsCollected, classPD)
print "after fit"

bc_svmobj = sc.broadcast(rbf_svm)

results = datalp.map(lambda x: bc_svmobj.value.predict(x))
print results.take(5)
