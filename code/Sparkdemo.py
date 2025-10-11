# Simple Spark demo to read/write Parquet files to HDFS

from pyspark.sql import SparkSession, functions as F

# HDFS base path (adjust username if needed)
HDFS_BASE = "hdfs://archmaster:9000"
OUT_PATH = f"{HDFS_BASE}/out/test_parquet_hdfs"

# Initialize Spark session
spark = (
    SparkSession.builder
    .appName("hdfs-demo")
    .config("spark.hadoop.fs.defaultFS", "hdfs://archmaster:9000")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .getOrCreate()
)

print("\n=== Spark + HDFS Demo ===")
print("Spark version:", spark.version)
print("Master:", spark.sparkContext.master)
print("Default parallelism:", spark.sparkContext.defaultParallelism)
print("Writing to HDFS path:", OUT_PATH)

# 2. Create test DataFrame
df = spark.range(1_000_000).withColumn("x", (F.rand() * 100).cast("float"))

# Show sample
print("\n=== Sample Data ===")
df.show(5, truncate=False)

# 3. Basic aggregation
agg = df.agg(
    F.count("*").alias("n"),
    F.avg("x").alias("avg_x"),
    F.sum("id").alias("sum_id")
)
print("\n=== Aggregation ===")
agg.show(truncate=False)

# 4. Write Parquet to HDFS
print("\n=== Writing to HDFS... ===")
(
    df.repartition(8)
      .write.mode("overwrite")
      .option("compression", "zstd")
      .parquet(OUT_PATH)
)

print(f"✅ Successfully wrote Parquet to HDFS at {OUT_PATH}")

# 5. Read back from HDFS
print("\n=== Reading back from HDFS... ===")
df2 = spark.read.parquet(OUT_PATH)
print("✅ Successfully read from HDFS")

# Verify row counts match
print("\n=== Row Count Verification ===")
print("Rows written:", df.count(), " | Rows read:", df2.count())

# 6. Group-by check
print("\n=== Histogram buckets (x/10) ===")
sample = (
    df2.withColumn("bucket", (F.col("x") / 10).cast("int"))
       .groupBy("bucket").count()
       .orderBy("bucket")
)
sample.show(10, truncate=False)

spark.stop()
print("\n✅ HDFS test complete. Spark can read/write to HDFS correctly.")
