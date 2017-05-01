from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.util import MLUtils

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

model = GradientBoostedTrees.trainClassifier(parsedTrainData,
                                             categoricalFeaturesInfo={}, numIterations=3)

predictions = model.predict(parsedTestData.map(lambda x: x.features))
labelsAndPredictions = parsedTestData.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(parsedTestData.count())
print('Test Error = ' + str(testErr))
