from pyspark.sql import SparkSession, functions as F

spark = (
    SparkSession.builder
    .appName("fruit-rot-demo")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .getOrCreate()
)

print("\n=== Spark started ===")
print("Spark version:", spark.version)
print("Master:", spark.sparkContext.master)
print("Default parallelism:", spark.sparkContext.defaultParallelism)


df = spark.range(1_000_000).withColumn("x", (F.rand() * 100).cast("float"))
agg = df.agg(F.count("*").alias("n"), F.avg("x").alias("avg_x"), F.sum("id").alias("sum_id"))
print("\n=== Aggregation ===")
agg.show(truncate=False)

out_path = "out/test_parquet"
(
    df.repartition(8)                
      .write.mode("overwrite")
      .option("compression", "zstd")  
      .parquet(out_path)
)

df2 = spark.read.parquet(out_path)
print("\n=== Read-back check (should match row count) ===")
print("rows written:", df.count(), " | rows read:", df2.count())


sample = (
    df2.withColumn("bucket", (F.col("x")/10).cast("int"))
       .groupBy("bucket").count()
       .orderBy("bucket")
)
print("\n=== Histogram buckets (x/10) ===")
sample.show(10, truncate=False)

spark.stop()
print("\n=== Done ===")
