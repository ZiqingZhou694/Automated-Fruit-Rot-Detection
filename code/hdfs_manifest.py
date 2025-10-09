# Generates a CSV manifest of image paths and labels from HDFS.

from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col

spark = (
    SparkSession.builder
    .appName("GenerateHDFSManifest")
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000")
    .getOrCreate()
)

# Recursively read all image files under testing directory
data_path = "hdfs://localhost:9000/data/fruits/test/"
df = spark.read.format("binaryFile").option("recursiveFileLookup", "true").load(data_path).select("path")

df = df.withColumn(
    "label",
    when(col("path").like("%fresh%"), "fresh")
    .when(col("path").like("%rotten%"), "rotten")
    .otherwise("unknown")
)

# Write manifest to HDFS out folder
out_path = "/out/manifest"
df.write.csv(out_path, header=True, mode="overwrite")

print(f"✅ Manifest written to {out_path}")
spark.stop()
