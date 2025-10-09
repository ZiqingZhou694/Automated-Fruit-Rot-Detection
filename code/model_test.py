"""# @Author(s): Aidan Eiler, Ziqing Zhou
 @Created: 10/7/2025
 @Modified: 10/8/2025

 Purpose:
     Test the trained Random Forest model on a single image from URL or file path.
     Extracts features and predicts whether fruit is fresh or rotten.

 Input:
     Image URL or local file path (command line argument)
     ../out/model - trained model
 Output:
     Prediction (fresh/rotten) with confidence"""

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
import extract_features

MODEL_PATH = "./out/model"


def log(msg):  # printf but with more style
    print(f"\n=== {msg} ===")

def is_url(s: str) -> bool:
    return s.lower().startswith(("http://", "https://"))

def download_image(url: str) -> str:
    """Download image from URL with a browser-like User-Agent to avoid 403."""
    log("downloading image from URL...")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/124.0 Safari/537.36"
        },
    )
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    with contextlib.closing(urllib.request.urlopen(req, timeout=20)) as resp, open(tmp.name, "wb") as f:
        f.write(resp.read())
    return tmp.name


if len(sys.argv) < 2:
    print("ERROR: Please provide an image URL or local file path")
    print('Usage:  py code/model_test.py "<url-or-local-path>"')
    sys.exit(1)

src = sys.argv[1]
if is_url(src):
    image_path = download_image(src)
else:
    if not os.path.isfile(src):
        print(f"ERROR: file not found: {src}")
        sys.exit(1)
    image_path = src

# initialize spark
log("initializing spark...")
spark = (
    SparkSession.builder
    .appName("model_test")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .config("spark.hadoop.hadoop.home.dir", "C:/hadoop")
    .config("spark.hadoop.io.native.lib.available", "false")
    .config("spark.pyspark.python", sys.executable)
    .config("spark.pyspark.driver.python", sys.executable)
    .getOrCreate()
)

# download and extract features
# image_path = download_image(image_url)
log("extracting features...")
features = extract_features.extract_features(image_path)

if features is None:
    print("ERROR: could not extract features")
    spark.stop()
    sys.exit(1)

# convert to spark dataframe
schema = T.StructType([
    T.StructField("path", T.StringType(), False),
    T.StructField("features", T.ArrayType(T.FloatType()), False)
])

data_frame = spark.createDataFrame([(image_path, features)], schema=schema)
data_frame = data_frame.withColumn("features", F.udf(lambda arr: Vectors.dense(
    arr), VectorUDT())(F.col("features")))

# load model and predict
log("loading model and predicting...")
model = PipelineModel.load(MODEL_PATH)
prediction_df = model.transform(data_frame)

result = prediction_df.select("prediction", "probability").first()
prediction_idx = int(result["prediction"])
probabilities = result["probability"].toArray()

# map prediction to label
import json, os
labels_path = os.path.join(MODEL_PATH, "labels.json")
with open(labels_path) as f:
    labels = json.load(f)  # e.g. ['rotten','fresh'] or ['fresh','rotten']

predicted_label = labels[prediction_idx]
confidence = probabilities[prediction_idx] * 100


# display results
log("PREDICTION RESULT")
# print(f"Image: {image_url}")
print(f"Image: {image_path}")
print(f"Prediction: {predicted_label.upper()}")
print(f"Confidence: {confidence:.2f}%")
print(f"\nProbabilities:")
for i, lbl in enumerate(labels):
    print(f"  {lbl}: {probabilities[i]*100:.2f}%")

# cleanup
os.unlink(image_path)

spark.stop()
print("\nPROGRAM COMPLETE")
