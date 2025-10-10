# =============================
# Distributed 128-D Feature Extraction from HDFS images
# =============================
from pyspark.sql import SparkSession, functions as F, types as T
import numpy as np
import cv2

# =============================
# CONFIGURATION
# =============================
HDFS_BASE = "hdfs://archmaster:9000"
DATASET_PATHS = [f"{HDFS_BASE}/data/train/",
                 f"{HDFS_BASE}/data/test/"]
OUT_PARQUET = f"{HDFS_BASE}/out/features_parquet"

spark = (
    SparkSession.builder
    .appName("ExtractFeatures_HDFS_BinaryFile_128D")
    .config("spark.hadoop.fs.defaultFS", HDFS_BASE)
    .config("spark.executor.memory", "4g")
    .config("spark.driver.memory", "6g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

def log(msg):
    print(f"\n=== {msg} ===")

# =============================
# FEATURE EXTRACTION FUNCTION (HDFS bytes)
# =============================
def featurize_bytes(content):
    try:
        arr = np.frombuffer(content, np.uint8)
        im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if im is None:
            return [0.0]*128

        # --- Resize and trim borders ---
        im = cv2.resize(im, (224, 224))
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
            if bottom>top+40 and right>left+40:
                return cv2.resize(img_bgr[top:bottom+1, left:right+1], (224,224))
            return img_bgr
        im = trim_border(im)

        # --- Color spaces ---
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        Vn = clahe.apply(V)
        hsv = cv2.merge([H, S, Vn])
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
        Lc, Ac, Bc = cv2.split(lab)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # --- Contour mask ---
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
            if (mask>0).sum() < 0.02*mask.size:
                mask[:] = 255
        else:
            mask[:] = 255

        # --- Histograms ---
        def hist(arr, bins, rng, m=None):
            return cv2.calcHist([arr],[0],m,bins,rng).flatten().astype(np.float32)

        h_hist  = hist(hsv[:,:,0], [24], [0,180], mask)
        s_hist  = hist(hsv[:,:,1], [24], [0,256], mask)
        v_hist  = hist(hsv[:,:,2], [24], [0,256], mask)
        a_hist  = hist(Ac,         [16], [0,256], mask)
        b_hist  = hist(Bc,         [16], [0,256], mask)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx*gx + gy*gy).astype(np.float32)
        mvals = mag[mask.astype(bool)] if mask is not None else mag.ravel()
        if mvals.size == 0:
            g_hist = np.zeros(16, np.float32)
        else:
            g_hist, _ = np.histogram(mvals, bins=16, range=(0,255))
            g_hist = g_hist.astype(np.float32)

        # --- Statistics ---
        def mean_std(x, msk):
            vals = x[msk.astype(bool)] if msk is not None else x.ravel()
            if vals.size == 0: return 0.0, 0.0
            return float(vals.mean()), float(vals.std())
        h_mean, h_std = mean_std(H, mask)
        s_mean, s_std = mean_std(S, mask)
        v_mean, v_std = mean_std(Vn, mask)
        edge_density = float((edges>0).sum()) / float(edges.size)
        dark_ratio   = float(((Vn<60) & (S>40) & (mask>0)).sum()) / max(1.0, float((mask>0).sum()))
        stats = np.array([h_mean,h_std,s_mean,s_std,v_mean,v_std, edge_density, dark_ratio], np.float32)

        # --- Per-block L1 + global L2 ---
        def l1(x):
            s = float(x.sum())
            return (x/(s+1e-6)).astype(np.float32)

        feat = np.concatenate([l1(h_hist), l1(s_hist), l1(v_hist),
                               l1(a_hist), l1(b_hist), l1(g_hist), stats], axis=0).astype(np.float32)
        feat = np.pad(feat, (0, 128 - feat.shape[0]), constant_values=0.0) if feat.shape[0] < 128 else feat[:128]
        nrm = float(np.linalg.norm(feat))
        feat = (feat/max(nrm,1e-8)).astype(np.float32)
        return feat.tolist()
    except Exception:
        return [0.0]*128

featurize_udf = F.udf(featurize_bytes, T.ArrayType(T.FloatType()))

# =============================
# PROCESS TRAIN AND TEST
# =============================
all_dfs = []
for dataset_path in DATASET_PATHS:
    log(f"loading images from {dataset_path}")
    df = spark.read.format("binaryFile") \
        .option("recursiveFileLookup", "true") \
        .load(dataset_path) \
        .select("path", "content")

    df = df.withColumn("features", featurize_udf(F.col("content"))) \
           .withColumn("label", F.when(F.col("path").like("%fresh%"), "fresh")
                               .when(F.col("path").like("%rotten%"), "rotten")
                               .otherwise("unknown")) \
           .withColumn("split", F.when(F.col("path").contains("/test/"), "test")
                                  .otherwise("train")) \
           .select("path", "label", "split", "features")

    all_dfs.append(df)

feat_df = all_dfs[0].unionByName(all_dfs[1])

log("writing Parquet to HDFS")
feat_df.repartition(8).write.mode("overwrite").partitionBy("split","label").parquet(OUT_PARQUET)

print(f"[OK] wrote Parquet to {OUT_PARQUET}")
spark.stop()