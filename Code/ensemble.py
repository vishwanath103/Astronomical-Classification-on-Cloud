from pyspark import SparkContext
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import BinaryClassificationMetrics

sc = SparkContext("local", "test_script")
data = sc.textFile("Training_Data.csv")

#Remove Header
header = data.first()
data = data.filter(lambda line: line != header)

trianData1,trianData2,trianData3,trianData4,trianData5,trianData6,trianData7,testData = data.randomSplit([0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125])

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

parsedTrainData1 = trianData1.map(parsePoint)
parsedTrainData2 = trianData2.map(parsePoint)
parsedTrainData3 = trianData3.map(parsePoint)
parsedTrainData4 = trianData4.map(parsePoint)
parsedTrainData5 = trianData5.map(parsePoint)
parsedTrainData6 = trianData6.map(parsePoint)
parsedTrainData7 = trianData7.map(parsePoint)

parsedTestData = testData.map(parsePoint)

model1 = SVMWithSGD.train(parsedTrainData1, iterations=100)
model2 = SVMWithSGD.train(parsedTrainData2, iterations=100)
model3 = SVMWithSGD.train(parsedTrainData3, iterations=100)
model4 = SVMWithSGD.train(parsedTrainData4, iterations=100)
model5 = SVMWithSGD.train(parsedTrainData5, iterations=100)
model6 = SVMWithSGD.train(parsedTrainData6, iterations=100)
model7 = SVMWithSGD.train(parsedTrainData7, iterations=100)

labelsAndPreds = parsedTestData.map(lambda p: (p.label, [model1.predict(p.features),model2.predict(p.features),model3.predict(p.features)
                                                          ,model4.predict(p.features),model5.predict(p.features),model6.predict(p.features),
                                                          model7.predict(p.features)]))

def parseResult(t):
    if t[1].count(0) > t[1].count(1):
        return (t[0],0)
    else:
        return (t[0],1)

parsedResult = labelsAndPreds.map(parseResult)
print parsedResult.take(5)

trainErr = parsedResult.filter(lambda (v, p): v == p).count() / float(parsedTestData.count())
accuracy = (trainErr) * 100
print("Accuracy = " + str(accuracy))
