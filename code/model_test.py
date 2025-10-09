# Purpose:
#     Test the trained Random Forest model stored in HDFS on a single image from URL.
#     Extracts features locally, then predicts via Spark model.

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml import PipelineModel
from pyspark.ml.linalg import Vectors, VectorUDT
import os, sys, urllib.request, contextlib, tempfile
import sys
import numpy as np
import cv2
import urllib.request
import tempfile
import os

# NOTE: Change this to match where your model is stored in HDFS
MODEL_PATH = "hdfs://localhost:9000/out/model"

def log(msg):
    print(f"\n=== {msg} ===")

def download_image(url):
    """Download image from URL."""
    log("Downloading image from URL...")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    urllib.request.urlretrieve(url, temp_file.name)
    return temp_file.name

def extract_features(image_path):
    """Extract 128-D feature vector from image."""
    im = cv2.imread(image_path)
    if im is None:
        return None

        # 0) resize
        im = cv2.resize(im, (224, 224))

    # HSV histograms
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()

    # Gradient magnitude histogram
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    grad_hist, _ = np.histogram(mag.ravel(), bins=32, range=(0, 255))

    # Concatenate and normalize
    feat = np.concatenate([h, s, v, grad_hist.astype(np.float32)]).astype(np.float32)
    nrm = np.linalg.norm(feat)
    return (feat / nrm).tolist() if nrm > 1e-8 else [0.0] * 128

# --- MAIN EXECUTION ---
if len(sys.argv) < 2:
    print("ERROR: Please provide an image URL")
    print("Usage: spark-submit model_test_hdfs.py <image_url>")
    sys.exit(1)

image_url = sys.argv[1]

# initialize spark
log("Initializing Spark session...")
spark = (
    SparkSession.builder
    .appName("model_test_hdfs")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

# download and extract features
image_path = download_image(image_url)
log("Extracting features...")
features = extract_features(image_path)

if features is None:
    print("ERROR: could not extract features")
    spark.stop()
    sys.exit(1)

# convert to Spark DataFrame
schema = T.StructType([
    T.StructField("path", T.StringType(), False),
    T.StructField("features", T.ArrayType(T.FloatType()), False)
])

data_frame = spark.createDataFrame([(image_path, features)], schema=schema)
data_frame = data_frame.withColumn(
    "features", F.udf(lambda arr: Vectors.dense(arr), VectorUDT())(F.col("features"))
)

# load model and predict
log("Loading model from HDFS and predicting...")
model = PipelineModel.load(MODEL_PATH)
prediction_df = model.transform(data_frame)

result = prediction_df.select("prediction", "probability").first()
prediction_idx = int(result["prediction"])
probabilities = result["probability"].toArray()

# map prediction to label
label_map = {0: "rotten", 1: "fresh"}
predicted_label = label_map.get(prediction_idx, "unknown")
confidence = probabilities[prediction_idx] * 100

# display results
log("PREDICTION RESULT")
print(f"Image: {image_url}")
print(f"Prediction: {predicted_label.upper()}")
print(f"Confidence: {confidence:.2f}%")
print(f"\nProbabilities:")
print(f"  Fresh:  {probabilities[1]*100:.2f}%")
print(f"  Rotten: {probabilities[0]*100:.2f}%")

# cleanup
os.unlink(image_path)
spark.stop()
print("\nPROGRAM COMPLETE")
