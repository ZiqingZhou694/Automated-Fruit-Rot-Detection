# Extract features from images stored in HDFS and save them as Parquet files.
# Input: images in HDFS (binaryFile format)
# Output: Parquet files with columns: path, label, split, features

from pyspark.sql import SparkSession, functions as F, types as T
import numpy as np
import cv2

# =============================
# CONFIGURATION
# =============================
HDFS_BASE = "hdfs://localhost:9000/"
DATASET_PATH = f"{HDFS_BASE}/data/fruits/test/"
OUT_PARQUET = f"{HDFS_BASE}out/features_parquet"

# =============================
# INITIALIZE SPARK
# =============================
spark = (
    SparkSession.builder
    .appName("ExtractFeatures_HDFS_BinaryFile")
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000")
    .config("spark.executor.memory", "4g")
    .config("spark.driver.memory", "6g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

def log(msg):
    print(f"\n=== {msg} ===")

# =============================
# 1. Read all images from HDFS recursively
# =============================
log("loading images from HDFS")
# binaryFile automatically loads paths and bytes
images_df = (
    spark.read.format("binaryFile")
    .option("recursiveFileLookup", "true")
    .load(DATASET_PATH)
    .select("path", "content")
)

log(f"found {images_df.count()} images")

# =============================
# 2. Feature extraction UDF
# =============================
def featurize_bytes(content):
    try:
        # Convert bytes to numpy array
        arr = np.frombuffer(content, np.uint8)
        im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if im is None:
            return [0.0] * 128

        im = cv2.resize(im, (224, 224))
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
        s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
        v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx*gx + gy*gy)
        lbp_hist, _ = np.histogram(mag.ravel(), bins=32, range=(0, 255))

        feat = np.concatenate([h, s, v, lbp_hist.astype(np.float32)]).astype(np.float32)
        nrm = np.linalg.norm(feat)
        return (feat / nrm).tolist() if nrm > 1e-8 else [0.0]*128
    except Exception:
        return [0.0]*128

featurize_udf = F.udf(featurize_bytes, T.ArrayType(T.FloatType()))

# =============================
# 3. Add features and label
# =============================
log("extracting features")
feat_df = (
    images_df.withColumn("features", featurize_udf(F.col("content")))
             .withColumn(
                 "label",
                 F.when(F.col("path").like("%fresh%"), "fresh")
                  .when(F.col("path").like("%rotten%"), "rotten")
                  .otherwise("unknown")
             )
             .withColumn(
                 "split",
                 F.when(F.col("path").contains("/train/"), "train")
                  .otherwise("test")
             )
             .select("path", "label", "split", "features")
)

# =============================
# 4. Write Parquet to HDFS
# =============================
log("writing Parquet to HDFS")
(
    feat_df.repartition(8)
           .write.mode("overwrite")
           .partitionBy("split", "label")
           .parquet(OUT_PARQUET)
)

print(f"[OK] wrote Parquet to {OUT_PARQUET}")
spark.stop()