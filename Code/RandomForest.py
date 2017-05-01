from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import BinaryClassificationMetrics

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

model = RandomForest.trainClassifier(parsedTrainData, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=21, featureSubsetStrategy="auto",
                                     impurity='entropy', maxDepth=20, maxBins=32)

predictions = model.predict(parsedTestData.map(lambda x: x.features))

labelsAndPredictions = parsedTestData.map(lambda lp: lp.label).zip(predictions)

tn = labelsAndPredictions.filter(lambda (v, p): v != p and p == 0 and v == 1).count()
tp = labelsAndPredictions.filter(lambda (v, p): v == p and v == 1).count()
fn = labelsAndPredictions.filter(lambda (v, p): v != p and p == 1 and v == 0).count()
fp = labelsAndPredictions.filter(lambda (v, p): v == p and p == 0).count()

print '7NN results:', 'tp='+str(tp), 'fp='+str(fp), 'tn='+str(tn), 'fn='+str(fn)

prec = tp / (tp+fp)
print "precision:"+str(prec)

testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(parsedTestData.count())
accuracy = (1-testErr) * 100
print('Test Error = ' + str(accuracy))
