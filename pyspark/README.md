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
*Action*: operations as reduce, count, show
*DAG (Directed Acyclic Graph)*: scheduling layer of Spark Architecture

Transformations are only added to a DAG of computation (**no data** has been loaded yet) and only when an action is called the DAG gets executed.
The aim of lazy evaluation is to optimize the data processing workflow:
Spark can make many optimization decisions (such as data partition) after it had a chance to look at the DAG in entirety.
This would not be possible if it executed everything as soon as it got it.

In alternative, the execution of transformations will materialize the many intermediate datasets in memory.
This is evidently not efficient and effective since you're really not interested in those intermediate results as such (those are just convenient abstractions while writing the program).
So, just tell Spark what is the eventual answer to get and it figures out best way to get there.
