from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils

sc = SparkContext("local", "test_script")
trainingFile = "sample.csv"
data = sc.textFile(trainingFile)

def parseTestData(line):
    attlist = line.split(",")
    print attlist[0]
    label = int(attlist[0])
    features = [float(attlist[2]) - float(attlist[4]),
                float(attlist[5]) - float(attlist[4]),
                float(attlist[4]) - float(attlist[3]),
                float(attlist[3]) - float(attlist[6])]
    return (label, features)

parsedTestData = data.map(parseTestData)
parsedTestData.saveAsTextFile("/media/vishwanath/Vishwanath/IIITB/Sem_2/PE/Spark/try_result")
