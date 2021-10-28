from pyspark.sql import SparkSession
from pyspark import SparkFiles
from pyspark.sql.functions import when

# create spark session
sc = SparkSession.builder.appName("PySparkExample")\
    .config("spark.sql.shuffle.partitions", "50") \
    .config("spark.driver.maxResultSize", "5g") \
    .config("spark.sql.execution.arrow.enabled", "true")\
    .getOrCreate()

dataframe = sc.read.json('nyt2.json') # kaggle data - New York Times Best Sellers
dataframe.show(10)
dataframe.head()

dataframe_dropdup = dataframe.dropDuplicates()
dataframe_dropdup.show(10)

dataframe.columns
dataframe.count()
dataframe.describe().show()

#################  Data Structures  ################
# Converting dataframe into an RDD
rdd_convert = dataframe.rdd

# Obtaining contents of df as Pandas dataFrame
dataframe.toPandas()

# repartition in rdd
dataframe.repartition(10).rdd.getNumPartitions()
dataframe.coalesce(1).rdd.getNumPartitions()


################### DataFrame API ###################
# select
dataframe.select("author").show(10)
dataframe.select("author", "title", "rank", "price").show(10)

# when
dataframe.select("title", when(dataframe.title != 'ODD HOURS', 1).otherwise(0)).show(10)

# isin
dataframe[dataframe.author.isin("John Sandford", "Emily Giffin")].show(5) # records with specified authors if in the given options

# like
dataframe.select("author", "title", dataframe.title.like("% THE %")).show(15)

# startswith - endswith
dataframe.select("author", "title", dataframe.title.startswith("THE")).show(5)
dataframe.select("author", "title", dataframe.title.endswith("NT")).show(5)

# substring
dataframe.select(dataframe.author.substr(1, 6).alias("title")).show()

# groupby
dataframe.groupBy("rank_last_week").count().show(10)


##################### SQL syntax ##################
# Registering a table
dataframe.registerTempTable("df")

sc.sql("select * from df").show(3)


sc.sql("select \
           CASE WHEN description LIKE '%love%' THEN 'Love_Theme' \
           WHEN description LIKE '%hate%' THEN 'Hate_Theme' \
           WHEN description LIKE '%happy%' THEN 'Happiness_Theme' \
           WHEN description LIKE '%anger%' THEN 'Anger_Theme' \
           WHEN description LIKE '%horror%' THEN 'Horror_Theme' \
           WHEN description LIKE '%death%' THEN 'Criminal_Theme' \
           WHEN description LIKE '%detective%' THEN 'Mystery_Theme' \
           ELSE 'Other_Themes' \
           END Themes \
   from df").groupBy('Themes').count().show()


###################### save result ##################
dataframe.select("author", "title").write.save("Authors_Titles.json", format="json")

# end session
sc.stop()
