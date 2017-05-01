from pyspark import SparkContext
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint

sc = SparkContext("local", "test_script")
trainingFile = "gs://vishu/TrainingData/Training_Data.csv"
data = sc.textFile(trainingFile)

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

labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())
print("Training Error = " + str(trainErr))

def getResults(t):
    result = ""
    if t[1] == 0:
        return "Non Quosar"
    else:
        return "Quosar"


resultList = labelsAndPreds.map(getResults)
resultList.saveAsTextFile("gs://vishu/SVMResult")
print resultList.take(5)
