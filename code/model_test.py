# model_test_hdfs.py
# Purpose:
#   Test the trained Random Forest model stored in HDFS on a single image (URL or local path)
#   using the latest 128-D feature extraction.

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml import PipelineModel
from pyspark.ml.linalg import Vectors, VectorUDT
import os, sys, urllib.request, contextlib, tempfile
import numpy as np
import cv2
import json

# =============================
# CONFIGURATION
# =============================
MODEL_PATH = "hdfs://archmaster:9000/out/model"

# =============================
# LOGGING
# =============================
def log(msg):
    print(f"\n=== {msg} ===")

# =============================
# URL UTILITIES
# =============================
def is_url(s: str) -> bool:
    return s.lower().startswith(("http://", "https://"))

def download_image(url: str) -> str:
    """Download image from URL with a browser-like User-Agent."""
    log("Downloading image from URL...")
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

# =============================
# FEATURE EXTRACTION (128-D, latest)
# =============================
def extract_features(image_path: str):
    """Extract 128-D feature vector with latest improvements."""
    try:
        im = cv2.imread(image_path)
        if im is None:
            return [0.0]*128

        # 0) resize
        im = cv2.resize(im, (224, 224))

        # --- border trim ---
        def trim_border(img_bgr, max_trim=20):
            h, w = img_bgr.shape[:2]
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            th_lo, th_hi = 8, 247
            top = 0
            while top < min(max_trim, h//4) and (gray[top,:].min()>th_hi-8 or gray[top,:].max()<th_lo+8):
                top += 1
            bottom = h-1
            while bottom > h-1-min(max_trim,h//4) and (gray[bottom,:].min()>th_hi-8 or gray[bottom,:].max()<th_lo+8):
                bottom -= 1
            left = 0
            while left < min(max_trim,w//4) and (gray[:,left].min()>th_hi-8 or gray[:,left].max()<th_lo+8):
                left += 1
            right = w-1
            while right > w-1-min(max_trim,w//4) and (gray[:,right].min()>th_hi-8 or gray[:,right].max()<th_lo+8):
                right -= 1
            if bottom>top+40 and right>left+40:
                return cv2.resize(img_bgr[top:bottom+1, left:right+1], (224,224))
            return img_bgr
        im = trim_border(im)

        # 1) color spaces & CLAHE
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        Vn = clahe.apply(V)
        hsv = cv2.merge([H,S,Vn])
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
        _, Ac, Bc = cv2.split(lab)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # 2) contour mask
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
            cv2.drawContours(mask,[hull],-1,255,-1)
            if (mask>0).sum() < 0.02*mask.size: mask[:]=255
        else:
            mask[:]=255

        # 3) histograms
        def hist(arr, bins, rng, m=None):
            return cv2.calcHist([arr],[0],m,bins,rng).flatten().astype(np.float32)
        h_hist  = hist(hsv[:,:,0],[24],[0,180],mask)
        s_hist  = hist(hsv[:,:,1],[24],[0,256],mask)
        v_hist  = hist(hsv[:,:,2],[24],[0,256],mask)
        a_hist  = hist(Ac,[16],[0,256],mask)
        b_hist  = hist(Bc,[16],[0,256],mask)
        gx = cv2.Sobel(gray, cv2.CV_32F,1,0,ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F,0,1,ksize=3)
        mag = np.sqrt(gx*gx+gy*gy).astype(np.float32)
        mvals = mag[mask.astype(bool)] if mask is not None else mag.ravel()
        g_hist = np.histogram(mvals, bins=16, range=(0,255))[0].astype(np.float32) if mvals.size>0 else np.zeros(16,np.float32)

        # 4) statistics
        def mean_std(x, msk):
            vals = x[msk.astype(bool)] if msk is not None else x.ravel()
            return (float(vals.mean()), float(vals.std())) if vals.size>0 else (0.0,0.0)
        h_mean,h_std = mean_std(H,mask)
        s_mean,s_std = mean_std(S,mask)
        v_mean,v_std = mean_std(Vn,mask)
        edge_density = float((edges>0).sum())/edges.size
        dark_ratio   = float(((Vn<60)&(S>40)&(mask>0)).sum())/max(1.0,(mask>0).sum())
        stats = np.array([h_mean,h_std,s_mean,s_std,v_mean,v_std,edge_density,dark_ratio],np.float32)

        # 5) concat & normalize
        def l1(x): s=float(x.sum()); return (x/(s+1e-6)).astype(np.float32)
        feat = np.concatenate([l1(h_hist),l1(s_hist),l1(v_hist),l1(a_hist),l1(b_hist),l1(g_hist),stats],axis=0).astype(np.float32)
        if feat.shape[0]!=128:
            feat = np.pad(feat,(0,128-feat.shape[0]),constant_values=0.0)
        feat /= max(np.linalg.norm(feat),1e-8)
        return feat.tolist()
    except Exception:
        return [0.0]*128

# =============================
# MAIN
# =============================
if len(sys.argv)<2:
    print("Usage: spark-submit model_test_hdfs.py <image_url_or_local_path>")
    sys.exit(1)

src = sys.argv[1]
if is_url(src):
    image_path = download_image(src)
else:
    if not os.path.isfile(src):
        print(f"ERROR: file not found: {src}")
        sys.exit(1)
    image_path = src

# initialize Spark
log("Initializing Spark session...")
spark = (
    SparkSession.builder
    .appName("model_test_hdfs")
    .config("spark.driver.memory", "4g")
    .config("spark.hadoop.fs.defaultFS","hdfs://archmaster:9000")
    .config("spark.pyspark.python", sys.executable)
    .config("spark.pyspark.driver.python", sys.executable)
    .getOrCreate()
)

# extract features
log("Extracting features...")
features = extract_features(image_path)
if features is None:
    print("ERROR: feature extraction failed")
    spark.stop()
    sys.exit(1)

# convert to DataFrame
schema = T.StructType([
    T.StructField("path",T.StringType(),False),
    T.StructField("features",T.ArrayType(T.FloatType()),False)
])
data_frame = spark.createDataFrame([(image_path, features)], schema=schema)
data_frame = data_frame.withColumn("features", F.udf(lambda arr: Vectors.dense(arr), VectorUDT())(F.col("features")))

# load model and predict
log("Loading model from HDFS and predicting...")
model = PipelineModel.load(MODEL_PATH)
prediction_df = model.transform(data_frame)

result = prediction_df.select("prediction", "probability").first()
prediction_idx = int(result["prediction"])
probabilities = result["probability"].toArray()

# --- map prediction using labels saved via Spark text() ---
labels_txt_path = MODEL_PATH + "/labels.txt"  # previously written via labels_df.write.text(...)
labels_df = spark.read.text(labels_txt_path)
labels = [row.value for row in labels_df.collect()]  # each row.value is a class label string

predicted_label = labels[prediction_idx]
confidence = probabilities[prediction_idx] * 100

# display results
log("PREDICTION RESULT")
print(f"Prediction: {predicted_label.upper()}")
print(f"Confidence: {confidence:.2f}%")
print(f"\nProbabilities:")
for i, lbl in enumerate(labels):
    print(f"  {lbl}: {probabilities[i]*100:.2f}%")

# cleanup
os.unlink(image_path)
spark.stop()
print("\nPROGRAM COMPLETE")
