from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col

spark = (
    SparkSession.builder
    .appName("GenerateHDFSManifest")
    .config("spark.hadoop.fs.defaultFS", "hdfs://archmaster:9000")
    .getOrCreate()
)

# Define the datasets
datasets = ["train", "test"]

for dataset in datasets:
    data_path = f"hdfs://archmaster:9000/data/{dataset}/"
    out_path = f"hdfs://archmaster:9000/out/manifest_{dataset}"

    # Recursively read all image files
    df = spark.read.format("binaryFile").option("recursiveFileLookup", "true").load(data_path).select("path")

    # Infer labels from path
    df = df.withColumn(
        "label",
        when(col("path").like("%fresh%"), "fresh")
        .when(col("path").like("%rotten%"), "rotten")
        .otherwise("unknown")
    )

    # Write CSV to HDFS
    df.write.csv(out_path, header=True, mode="overwrite")
    print(f"✅ Manifest for {dataset} written to {out_path}")

spark.stop()