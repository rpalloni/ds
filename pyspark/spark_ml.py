from pyspark import SparkContext
from pyspark import SparkFiles
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml.feature import VectorAssembler
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
ss = SparkSession.builder.getOrCreate()

url = 'https://raw.githubusercontent.com/rpalloni/dataset/master/bankruptcy.txt'

sc.addFile(url)

rdd = sc.textFile("file://"+SparkFiles.get('bankruptcy.txt')).map(lambda n: n.split(","))
type(rdd)
rdd.getNumPartitions()

# Print the RDD content
rdd.count()
print(rdd.collect())

for i in rdd.take(5):
    print(i)

d = rdd.toDF()
print(d.show())

# RDD to DataFrame
schema = StructType([
    StructField('IndRisk', StringType(), True),
    StructField('MngRisk', StringType(), True),
    StructField('FinFlex', StringType(), True),
    StructField('Cred', StringType(), True),
    StructField('Compet', StringType(), True),
    StructField('OpRisk', StringType(), True),
    StructField('Class', StringType(), True)
])

df = ss.createDataFrame(rdd, schema)
df.printSchema()
df.show()

# Logistic Regression
predictors_assembler = VectorAssembler(
    inputCols=['IndRisk', 'MngRisk', 'FinFlex', 'Cred', 'Compet', 'OpRisk'],
    outputCol='predictors')

dfl = predictors_assembler.transform(df)
dfl.show()

train_data, test_data = dfl.randomSplit([0.75, 0.25])
lm = LogisticRegression(featuresCol='predictors', labelCol='Class')

res = lm.fit(train_data)
res.coefficients
res.intercept
res.summary.residuals.show()

pred = res.evaluate(test_data)

# end session
ss.stop()
