from pyspark import SparkContext
from pyspark import SparkFiles
from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.clustering import KMeans, KMeansModel

'''
P=positive, A=Average, N=Negative, B=Bankruptcy, NB=NonBankruptcy
1 - Industrial Risk: {P, A, N}
2 - Management Risk: {P, A, N}
3 - Fiancial Flexibility: {P, A, N}
4 - Credibility: {P, A, N}
5 - Competitiveness: {P, A, N}
6 - Operating Risk: {P, A, N}
7 - Class {B, NB}
'''

sc = SparkContext(appName="Bank")

url = 'https://raw.githubusercontent.com/rpalloni/dataset/master/bankruptcy.txt'

sc.addFile(url)

data = sc.textFile("file://"+SparkFiles.get('bankruptcy.txt')) # RDD
data.count()



