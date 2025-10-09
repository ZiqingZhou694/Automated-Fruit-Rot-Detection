# Check how many feature vectors are all zeros in a Parquet dataset stored in HDFS.
from pyspark.sql import SparkSession, functions as F

# Initialize Spark session
spark = (
    SparkSession.builder
    .appName("ParquetZeroCheck")
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000")
    .getOrCreate()
)

# Path to features parquet in HDFS
PARQUET_PATH = "hdfs://localhost:9000/out/features_parquet"

# Load parquet dataset
df = spark.read.parquet(PARQUET_PATH)

# Compute whether each feature vector sums to zero
# We use aggregate + sum built-in SQL function instead of Python lambda
df = df.withColumn("sum_features", F.aggregate(F.col("features"), F.lit(0.0), lambda acc, x: acc + x))

# Count zero and total
zero = df.filter(F.col("sum_features") == 0.0).count()
total = df.count()

print(f"total = {total}, zero = {zero}, zero_ratio = {zero/total if total > 0 else 0:.4f}")

# Group by label and count zero-feature rows per label
df.groupBy("label").agg(
    F.count("*").alias("cnt"),
    F.sum(F.when(F.col("sum_features") == 0.0, 1).otherwise(0)).alias("zero_in_label")
).show()

spark.stop()
