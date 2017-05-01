from pyspark import SparkContext

sc = SparkContext(appName="PysparkKnearestNeigbours")
records = sc.textFile("sample.csv")
recordlist = records.collect()

testset,trainingset = records.randomSplit([1,2])

def knn(testinstance):
    nearestNeigbours = trainingset.cartesian(testinstance) \
    .map(lambda (training,test):(training, distanceAbs(training, test, 3))) \
    .sortBy(lambda (trainingInstance, distance):distance) \
    .take(5)
    return nearestNeigbours
