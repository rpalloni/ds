from pyspark import SparkContext
from pyspark.mllib.stat import Statistics

# create spark session
sc = SparkContext(appName="Stats")


inputData = [
    [2, 2, 3],
    [0, 9, 2],
    [4, 4, 4]
]

matrix = sc.parallelize(inputData) # RDD

summary = Statistics.colStats(matrix)
print(summary.mean()) # [2 5 3]

seriesx = sc.parallelize([41, 36, 12, 18, 28, 23, 19, 8, 7, 16, 11, 14, 18, 14, 34, 6, 30, 11, 1, 4, 32, 23, 45, 18, 26, 32, 41, 2, 29, 30])
seriesy = sc.parallelize([190, 118, 149, 313, 300, 99, 19, 194, 256, 65, 334, 307, 78, 322, 44, 8, 320, 25, 95, 25, 63, 51, 89, 48, 26, 99, 210, 280, 32, 65])

c = Statistics.corr(seriesx, seriesy, method='pearson')
print(c)
