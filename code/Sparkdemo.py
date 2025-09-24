from pyspark.sql import SparkSession, functions as F

# 1) 启动本机多核 Spark，会自动用你 CPU 的所有核心
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

# 2) 造一些测试数据（1,000,000 行），做个简单聚合
df = spark.range(1_000_000).withColumn("x", (F.rand() * 100).cast("float"))
agg = df.agg(F.count("*").alias("n"), F.avg("x").alias("avg_x"), F.sum("id").alias("sum_id"))
print("\n=== Aggregation ===")
agg.show(truncate=False)

# 3) 写入 Parquet 再读回来（验证 I/O、压缩、分区）
out_path = "out/test_parquet"
(
    df.repartition(8)                 # 模拟并行写
      .write.mode("overwrite")
      .option("compression", "zstd")  # 压缩更快更省
      .parquet(out_path)
)

df2 = spark.read.parquet(out_path)
print("\n=== Read-back check (should match row count) ===")
print("rows written:", df.count(), " | rows read:", df2.count())

# 4) 简单转换再看几行
sample = (
    df2.withColumn("bucket", (F.col("x")/10).cast("int"))
       .groupBy("bucket").count()
       .orderBy("bucket")
)
print("\n=== Histogram buckets (x/10) ===")
sample.show(10, truncate=False)

spark.stop()
print("\n=== Done ===")
