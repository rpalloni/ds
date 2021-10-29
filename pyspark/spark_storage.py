# StorageLevel decides whether RDD should be stored in-memory or to disk (or both)
from pyspark import SparkContext, StorageLevel

sc = SparkContext(appName="StorageLevel")

words = ('apple', 'lenovo', 'dell', 'samsung')
rdd = sc.parallelize(words)
rdd.persist(StorageLevel.MEMORY_AND_DISK_2) # store in memory AND disk with 2x replication
rdd.count() # open tab Storage in Spark Web UI
