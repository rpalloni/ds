# PySpark

Spark Data Processing Framework

Ecosystem components:
* core
* SQL
* streaming
* MLlib

Docs: http://spark.apache.org/docs/latest/api/python/index.html


### Spark data structure interfaces
- RDD (*original data structure for Apache Spark*)
- DataFrame (*Python and R like*)
- Datasets (*Java and Scala only*)

### Data partitioning
Core Spark feature: data are split in subsets and distributed across nodes to optimize management and calculation

### In-memory computing
Data is stored in RAM to access the data quickly and accelerate analytics

### Lazy evaluation of transformations
*Transformation*: map, filter, join, union \
*Action*: operations as reduce, count, show \
*DAG (Directed Acyclic Graph)*: scheduling layer of Spark Architecture

Transformations are only added to a DAG of computation (**no data** has been loaded yet) and only when an action is called the DAG gets executed.
The aim of lazy evaluation is to optimize the data processing workflow:
Spark can make many optimization decisions (such as data partition) after it had a chance to look at the DAG in entirety.
This would not be possible if it executed everything as soon as it got it.

In alternative, the execution of transformations will materialize the many intermediate datasets in memory.
This is evidently not efficient and effective since you're really not interested in those intermediate results as such (those are just convenient abstractions while writing the program).
So, just tell Spark what is the eventual answer to get and it figures out best way to get there.

### Example of transformation pipelining
A series of transformation on intermediate (abstract) datasets: \
dt = dataframe.where(dataframe.price > 0).select('publisher', 'price').groupby('publisher').avg('price') # lazy evaluation \

The actual computation:
dt.show() # execution with action

### Web UI
https://spark.apache.org/docs/latest/web-ui.html
Apache Spark provides a web app with user interfaces:
* Jobs
* Stages
* Tasks
* Storage
* Environment
* Executors
* SQL
to monitor the status of the application, resource consumption of the cluster and configurations.
In a Spark local environment, components are accessible @:
* Web UI -> localhost:4040
* Resource Manager -> localhost:9870
* Job Tracker -> localhost:8088
* Node Info -> localhost:8042

~~~
from pyspark.sql import SparkSession

# create spark session
spark = SparkSession.builder.master("local[1]") \
    .appName('DataPlatform') \
    .getOrCreate()

# transformation
dataframe = (
    spark.read                      # job(0): read
    .option('inferSchema', 'True')  # job(1): infer
    .option('header', 'True')
    .csv('data.csv')
)

# action
dataframe.count()                   # job(2): count

spark.stop()
~~~

![webui](https://user-images.githubusercontent.com/17080117/165984346-502a0697-629b-40cd-ba14-0809b38d260d.png)
