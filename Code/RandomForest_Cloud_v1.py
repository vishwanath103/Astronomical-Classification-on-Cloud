from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils

sc = SparkContext("local", "test_script")
##data = sc.textFile("Training_Data.csv")
data = sc.textFile("gs://vishu/TrainingData/Training_Data.csv")

def parsePoint(line):
    attlist = line.split(",")
    label = 0
    if attlist[1] == "QSO":
        label = 1
    elif attlist[1] == "GALAXY":
        label = 2
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

model = RandomForest.trainClassifier(parsedData, numClasses=3, categoricalFeaturesInfo={},
                                     numTrees=21, featureSubsetStrategy="auto",
                                     impurity='entropy', maxDepth=20, maxBins=32)

predictions = model.predict(parsedData.map(lambda x: x.features))

labelsAndPredictions = parsedData.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(parsedData.count())
accuracy = (1-testErr) * 100
print('Accuracy = ' + str(accuracy))
