# pip3 install pyspark

from pyspark.sql import SparkSession
from pyspark import SparkFiles
from pyspark.sql.types import StructType, IntegerType, DoubleType, StringType

from pyspark.sql.functions import isnan, when, count, col
from pyspark.ml.feature import Imputer, VectorAssembler
from pyspark.ml.regression import LinearRegression

url = 'https://raw.githubusercontent.com/rpalloni/dataset/master/airquality.csv'

# create pyspark session
spark = SparkSession.builder.appName('Session01').getOrCreate()
spark

spark.sparkContext.addFile(url)

schema = StructType() \
            .add("Ozone", IntegerType(), True) \
            .add("SolarRay", IntegerType(), True) \
            .add("Wind", DoubleType(), True) \
            .add("Temp", IntegerType(), True) \
            .add("Month", IntegerType(), True) \
            .add("Day", IntegerType(), True)

df_s = spark.read.csv("file://"+SparkFiles.get('airquality.csv'), header=True, schema=schema) # header present in file

df_s
df_s.show()
df_s.count()
df_s.columns
df_s.select("Ozone").show(10)
df_s.describe(['Ozone', 'SolarRay']).show()

# NAs
df_s.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_s.columns]).show()
df_s.na.drop(how='any') # any/all, thresh (min # of null values to delete record)

# missing values
imputer = Imputer(
    inputCols=['Ozone', 'SolarRay'],
    outputCols=['{}_imputed'.format(c) for c in ['Ozone', 'SolarRay']]
).setStrategy('mean')

imputer.fit(df_s).transform(df_s).show()
df_s = imputer.fit(df_s).transform(df_s)

# filter
df_s.filter('Ozone>=120').show()
df_s.filter((df_s['Ozone'] >= 120) & (df_s['Wind'] == 4)).show()

# aggregate
df_s.groupBy('Month').count().show()
df_s.groupBy('Month').avg('Ozone').show()
df_s.groupBy('Month').max('Ozone').show()


# MLlib
predictors_assembler = VectorAssembler(
    inputCols=['SolarRay_imputed', 'Wind', 'Temp'],
    outputCol='predictors')

df_reg = predictors_assembler.transform(df_s).select('Ozone_imputed', 'predictors')
df_reg.show()

train_data, test_data = df_reg.randomSplit([0.75, 0.25])
lm = LinearRegression(featuresCol='predictors', labelCol='Ozone_imputed')

res = lm.fit(train_data)
res.coefficients
res.intercept

pred = res.evaluate(test_data)
pred.r2adj
pred.meanAbsoluteError
pred.meanSquaredError
pred.predictions.show()
