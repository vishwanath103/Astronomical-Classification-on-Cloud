from pyspark import SparkContext
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import BinaryClassificationMetrics

sc = SparkContext("local", "test_script")
data = sc.textFile("Training_Data.csv")

def parsePoint(line):
    attlist = line.split(",")
    label = 0
    if attlist[1] == "QSO":
        label = 1
    else:
        label = 0
    features = [float(attlist[2]) - float(attlist[4]),
                float(attlist[5]) - float(attlist[4]),
                float(attlist[4]) - float(attlist[3]),
                float(attlist[3]) - float(attlist[6])]
    return LabeledPoint(label, features)

#Remove Header
header = data.first()
data = data.filter(lambda line: line != header)
parsedData = data.map(parsePoint)


model = SVMWithSGD.train(parsedData, iterations=100)

labelsAndPreds = parsedData.map(lambda p: (model.predict(p.features),p.label))
trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())
print("Training Error = " + str(trainErr))

metrics = BinaryClassificationMetrics(labelsAndPreds)
print ("Precision = %s" %metrics.areaUnderPR)

def getResults(t):
    result = ""
    if t[1] == 0:
        return (t[0],"NQSO")
    else:
        return (t[0],"QSO")


resultList = labelsAndPreds.map(getResults)
##resultList.saveAsTextFile("/media/vishwanath/Vishwanath/IIITB/Sem_2/PE/Spark/Results")
print resultList.take(5)
