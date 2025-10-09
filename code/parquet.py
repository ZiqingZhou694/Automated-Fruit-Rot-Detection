""" 
Purpose:
     Read a manifest CSV file (path,label), compute handcrafted image features
    (HSV histograms + gradient magnitude histogram), normalize them,
     and save the results into a Parquet dataset.

 Input:
     out/manifest.csv   (columns: path, label)

 Output:
     out/features_parquet
     (columns: path, label, features[array<float>])

 Notes:
     - Uses OpenCV (cv2) to read and process images directly from disk.
     - Features are 128-dimensional vectors:
         * H histogram (32 bins)
         * S histogram (32 bins)
         * V histogram (32 bins)
         * Gradient magnitude histogram (32 bins)
     - Each feature vector is L2-normalized."""

import pathlib
from pyspark.sql import SparkSession, functions as F, types as T
import sys
import numpy as np
import cv2
import extract_features

OUT_PARQUET = "./out/features_parquet"


def log(msg):  # printf but with more style
    print(f"\n=== {msg} ===")


# -----------------------------------------------------------
# 1) Initialize Spark session
# -----------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("featrure_extraction")
    .master("local[*]")                          # Use all local cores
    .config("spark.driver.memory", "6g")          # Driver memory
    .config("spark.sql.shuffle.partitions", "8")  # Reduce shuffle partitions
    .config("spark.hadoop.hadoop.home.dir", "C:/hadoop")
    .config("spark.hadoop.io.native.lib.available", "false")
    # Force PySpark to use the same Python interpreter as current process
    .config("spark.pyspark.python", sys.executable)
    .config("spark.pyspark.driver.python", sys.executable)
    .getOrCreate()
)

# -----------------------------------------------------------
# 2) Load manifest (CSV with path,label)
# -----------------------------------------------------------
man = (
    spark.read.option("header", True)
    .csv("out/manifest.csv")
    .dropna(subset=["path", "label"])  # drop rows with missing path/label
)
log(f"manifest rows={man.count()}")

# -----------------------------------------------------------
# 3) Feature extraction UDF
# -----------------------------------------------------------

# Register UDF in Spark
module_path = pathlib.Path(__file__).resolve().parent / "extract_features.py"
spark.sparkContext.addPyFile(str(module_path))
featurize_udf = F.udf(extract_features.extract_features, T.ArrayType(T.FloatType()))

log("compute features (from disk)")
feat_df = (
    man.withColumn("features", featurize_udf(F.col("path")))
       .select("path", "label", "features")
)
# -----------------------------------------------------------
# 4) Add train/test split column
# -----------------------------------------------------------
log("add train/test split")
feat_df = feat_df.withColumn(
    "split",
    F.when(F.col("path").contains("\\train\\"), "train")
     .otherwise("test")
)

# -----------------------------------------------------------
# 5) Write Parquet output
# -----------------------------------------------------------
log("write parquet")
(
    feat_df.repartition(8)   # balance data across 8 partitions
    .write.mode("overwrite")
    # split directory by split first then label
          .partitionBy("split", "label")
          .parquet(OUT_PARQUET)
)

print(f"[OK] wrote parquet to: {OUT_PARQUET}")
spark.stop()
