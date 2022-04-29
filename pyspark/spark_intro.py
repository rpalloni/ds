from pyspark import StorageLevel
from pyspark.sql import SparkSession

# create spark session
spark = SparkSession.builder.master("local[1]") \
    .appName('DataPlatform') \
    .getOrCreate()

# open Web UI at localhost:4040

# transformation
dataframe = (
    spark.read                      # job(0): read
    .option('inferSchema', 'True')  # job(1): infer
    .option('header', 'True')
    .csv('data.csv')
)

# action
dataframe.count()                   # job(2): count

# persist
dataframe.persist(StorageLevel.MEMORY_AND_DISK)

spark.stop()
