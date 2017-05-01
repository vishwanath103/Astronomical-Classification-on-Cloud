from pyspark import SparkContext
from pyspark.sql import SQLContext
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import os

#os.chdir("/home/bailorevishwanath02")
sc = SparkContext(appName = "kNN")
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
autoDF = sqlContext.createDataFrame(autolp, ["g-r","u-r","r-i","i-z"])
(trainingData, testData) = autoDF.randomSplit([0.8,0.2])

df = trainingData.toPandas()
df2 = testData.toPandas()
nbrs = NearestNeighbors(n_neighbors=7, algorithm='auto').fit(df)

testingFile = "gs://vishu/TestingData/"
testData = sc.textFile(testingFile)

def transformTestData(inputStr):
    attlist = inputStr.split(",")
    data = (float(attlist[4]) - float(attlist[6]),
         float(attlist[7]) - float(attlist[6]),
         float(attlist[6]) - float(attlist[5]),
         float(attlist[5]) - float(attlist[8]))
    return data

testlp = testData.map(transformTestData)
testDF = sqlContext.createDataFrame(testlp, ["g-r","u-r","r-i","i-z"])
testDFpd = testDF.toPandas()
distances, indices = nbrs.kneighbors(testDFpd)
##print distances

predicted_class = []
for i in range(indices.size/7):
    quasar_count = 0
    non_quasar_count = 0
    for j in range(indices[i].size):
        if(classes[indices[i,j]] == 'QSO'):
            quasar_count += (1/(distances[i,j]*distances[i,j]))
        else:
            non_quasar_count += (1/(distances[i,j]*distances[i,j]))
    if(quasar_count > non_quasar_count):
        predicted_class.append('QSO')
    else:
        predicted_class.append('NQSO')

resultRdd = sc.parallelize(predicted_class)
resultRdd.saveAsTextFile("gs://vishu/Result")


