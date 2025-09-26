from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder.appName("zero-ratio").getOrCreate()
df = spark.read.parquet("C:/spark_out/features_parquet")


zero = df.where(F.aggregate("features", F.lit(0.0), lambda acc, x: acc + x) == 0.0).count()
total = df.count()
print(f"total = {total}, zero = {zero}, zero_ratio = {zero/total:.4f}")


df.groupBy("label").agg(
    F.count("*").alias("cnt"),
    F.sum(F.when(F.aggregate("features", F.lit(0.0), lambda acc, x: acc + x) == 0.0, 1).otherwise(0)).alias("zero_in_label")
).show()

spark.stop()
