from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split

spark = SparkSession.builder.appName("NetWordCount").getOrCreate()

# Create DataFrame with data received from localhost:9999
lines = spark.readStream.format("socket").option("host", "localhost").optioin("port", 9999).load()

# Split the lines into words
words = lines.select(explode(split(lines.value, " ")).alias("word"))

# Generate running word count
wordCounts = words.groupBy("word").count()

# Start the query that prints the running counts to the console
query = wordCounts.writeStream.outputMode("complete").format("console").start()

query.awaitTermination()