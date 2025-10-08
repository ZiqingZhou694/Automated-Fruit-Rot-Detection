# @Author(s): Aidan Eiler, Ziqing Zhou
# @Created: 10/7/2025
# @Modified: 10/7/2025
#
# Purpose:
#     Test the trained Random Forest model on a single image from URL or file path.
#     Extracts features and predicts whether fruit is fresh or rotten.

# Input:
#     Image URL or local file path (command line argument)
#     ../out/model - trained model
# Output:
#     Prediction (fresh/rotten) with confidence

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


# def download_image(url):
#     """Download image from URL."""
#     log("downloading image from URL...")
#     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
#     urllib.request.urlretrieve(url, temp_file.name)
#     return temp_file.name


# def extract_features(image_path):  # using extract_features.py as a template
#     """Extract 128-D feature vector from image."""
#     im = cv2.imread(image_path)
#     if im is None:
#         return None

#     im = cv2.resize(im, (224, 224))

#     # HSV histograms
#     hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
#     h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
#     s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
#     v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()

#     # Gradient magnitude histogram
#     gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
#     gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
#     mag = np.sqrt(gx*gx + gy*gy)
#     grad_hist, _ = np.histogram(mag.ravel(), bins=32, range=(0, 255))

#     # Concatenate and normalize
#     feat = np.concatenate(
#         [h, s, v, grad_hist.astype(np.float32)]).astype(np.float32)
#     nrm = np.linalg.norm(feat)
#     return (feat / nrm).tolist() if nrm > 1e-8 else [0.0]*128
"""
    Same extractor from extract_features.py
    """
def extract_features(image_path: str):
    
    try:
        im = cv2.imread(image_path)
        if im is None:
            return [0.0]*128

        # 0) resize
        im = cv2.resize(im, (224, 224))

        # --- border trim: cut constant margins (handles black/white frames) ---
        def trim_border(img_bgr, max_trim=20):
            h, w = img_bgr.shape[:2]
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            th_lo, th_hi = 8, 247
            top = 0
            while top < min(max_trim, h//4) and (gray[top,:].min()>th_hi-8 or gray[top,:].max()<th_lo+8):
                top += 1
            bottom = h-1
            while bottom > h-1-min(max_trim, h//4) and (gray[bottom,:].min()>th_hi-8 or gray[bottom,:].max()<th_lo+8):
                bottom -= 1
            left = 0
            while left < min(max_trim, w//4) and (gray[:,left].min()>th_hi-8 or gray[:,left].max()<th_lo+8):
                left += 1
            right = w-1
            while right > w-1-min(max_trim, w//4) and (gray[:,right].min()>th_hi-8 or gray[:,right].max()<th_lo+8):
                right -= 1
            if bottom>top+40 and right>left+40:  # keep reasonable size
                return cv2.resize(img_bgr[top:bottom+1, left:right+1], (224,224))
            return img_bgr

        im = trim_border(im)

        # 1) color spaces
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)

        # illumination normalization on V
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        Vn = clahe.apply(V)
        hsv = cv2.merge([H, S, Vn])

        lab = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
        Lc, Ac, Bc = cv2.split(lab)

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # 2) contour mask from edges (no color threshold)
        gb = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(gb, 50, 150)
        kernel = np.ones((5,5), np.uint8)
        edges = cv2.dilate(edges, kernel, 1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, 1)

        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        if cnts:
            big = max(cnts, key=cv2.contourArea)
            hull = cv2.convexHull(big)
            cv2.drawContours(mask, [hull], -1, 255, -1)
            # if too tiny area -> fallback to full image
            if (mask>0).sum() < 0.02*mask.size:
                mask[:] = 255
        else:
            mask[:] = 255

        # 3) histograms (masked)
        def hist(arr, bins, rng, m=None):
            return cv2.calcHist([arr],[0],m,bins,rng).flatten().astype(np.float32)

        h_hist  = hist(hsv[:,:,0], [24], [0,180], mask)              # 24
        s_hist  = hist(hsv[:,:,1], [24], [0,256], mask)              # 24
        v_hist  = hist(hsv[:,:,2], [24], [0,256], mask)              # 24
        a_hist  = hist(Ac,         [16], [0,256], mask)              # 16
        b_hist  = hist(Bc,         [16], [0,256], mask)              # 16

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx*gx + gy*gy).astype(np.float32)
        if mask is not None:
            mvals = mag[mask.astype(bool)]
        else:
            mvals = mag.ravel()
        if mvals.size == 0:
            g_hist = np.zeros(16, np.float32)
        else:
            g_hist, _ = np.histogram(mvals, bins=16, range=(0,255))
            g_hist = g_hist.astype(np.float32)                        # 16

        # 4) statistics (8 dims)
        def mean_std(x, msk):
            vals = x[msk.astype(bool)] if msk is not None else x.ravel()
            if vals.size == 0: return 0.0, 0.0
            return float(vals.mean()), float(vals.std())
        h_mean, h_std = mean_std(H, mask)
        s_mean, s_std = mean_std(S, mask)
        v_mean, v_std = mean_std(Vn, mask)

        edge_density = float((edges>0).sum()) / float(edges.size)
        dark_ratio   = float(((Vn<60) & (S>40) & (mask>0)).sum()) / max(1.0, float((mask>0).sum()))
        stats = np.array([h_mean,h_std,s_mean,s_std,v_mean,v_std, edge_density, dark_ratio], np.float32)  # 8

        # 5) per-block L1 -> concat -> global L2
        def l1(x):
            s = float(x.sum())
            return (x/(s+1e-6)).astype(np.float32)

        feat = np.concatenate([l1(h_hist), l1(s_hist), l1(v_hist),
                               l1(a_hist), l1(b_hist), l1(g_hist), stats], axis=0).astype(np.float32)
        # sanity to 128
        if feat.shape[0] != 128:
            if feat.shape[0] > 128:
                feat = feat[:128]
            else:
                feat = np.pad(feat, (0, 128-feat.shape[0]), constant_values=0.0)

        nrm = float(np.linalg.norm(feat))
        feat = (feat/max(nrm,1e-8)).astype(np.float32)
        return feat.tolist()
    except Exception:
        return [0.0]*128


# check for image URL argument
# if len(sys.argv) < 2:
#     print("ERROR: Please provide an image URL")
#     print("Usage: python model_test.py <image_url>")
#     sys.exit(1)

# image_url = sys.argv[1]

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
features = extract_features(image_path)

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
# i flipped this because when the model was trained with fresh=1, rotten=0 the results almost seemed to be the opposite. todo if need be
# label_map = {0: "rotten", 1: "fresh"}
# predicted_label = label_map.get(prediction_idx, "unknown")
# confidence = probabilities[prediction_idx] * 100

# modified
import json, os
labels_path = os.path.join(MODEL_PATH, "labels.json")
with open(labels_path) as f:
    labels = json.load(f)  # e.g. ['rotten','fresh'] or ['fresh','rotten']

predicted_label = labels[prediction_idx]
# print(f"  {labels[0]}: {probabilities[0]*100:.2f}%")
# print(f"  {labels[1]}: {probabilities[1]*100:.2f}%")
confidence = probabilities[prediction_idx] * 100


# display results
log("PREDICTION RESULT")
# print(f"Image: {image_url}")
print(f"Image: {image_path}")
print(f"Prediction: {predicted_label.upper()}")
print(f"Confidence: {confidence:.2f}%")
print(f"\nProbabilities:")
# print(f"  Fresh:  {probabilities[0]*100:.2f}%")
# print(f"  Rotten: {probabilities[1]*100:.2f}%")
for i, lbl in enumerate(labels):
    print(f"  {lbl}: {probabilities[i]*100:.2f}%")

# cleanup
os.unlink(image_path)

spark.stop()
print("\nPROGRAM COMPLETE")
