from pyspark import SparkContext
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.classification import DecisionTreeClassifier

sc = SparkContext("local", "test_script")
data = sc.textFile("Training_Data.csv")
trianData,testData = data.randomSplit([0.8,0.2])

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
header = trianData.first()
trianData = trianData.filter(lambda line: line != header)

header1 = testData.first()
testData = testData.filter(lambda line: line != header1)

parsedTrainData = trianData.map(parsePoint)
parsedTestData = testData.map(parsePoint)

model = SVMWithSGD.train(parsedTrainData, iterations=50)

labelsAndPreds = parsedTestData.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda (v, p): v == p).count() / float(parsedTestData.count())
accuracy = (trainErr) * 100
print("Accuracy = " + str(accuracy))

##metrics = BinaryClassificationMetrics(labelsAndPreds)
##print ("accuracy = %s" %metrics.areaUnderROC)

def getResults(t):
    result = ""
    if t[1] == 0:
        return (t[0],"NQSO")
    else:
        return (t[0],"QSO")


resultList = labelsAndPreds.map(getResults)
##resultList.saveAsTextFile("/media/vishwanath/Vishwanath/IIITB/Sem_2/PE/Spark/Results")
print resultList.take(5)
