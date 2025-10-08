# Purpose:
#     Read a manifest CSV file (path,label), compute handcrafted image features
#     (HSV histograms + gradient magnitude histogram), normalize them,
#     and save the results into a Parquet dataset.

# Input:
#     out/manifest.csv   (columns: path, label)

# Output:
#     out/features_parquet
#     (columns: path, label, features[array<float>])

# Notes:
#     - Uses OpenCV (cv2) to read and process images directly from disk.
#     - Features are 128-dimensional vectors:
#         * H histogram (32 bins)
#         * S histogram (32 bins)
#         * V histogram (32 bins)
#         * Gradient magnitude histogram (32 bins)
#     - Each feature vector is L2-normalized.

from pyspark.sql import SparkSession, functions as F, types as T
import sys
import numpy as np
import cv2

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


# def featurize_path(p: str):

#     # Load image from disk and compute a 128-D feature vector:
#     # - 32-bin histogram of Hue (HSV)
#     # - 32-bin histogram of Saturation (HSV)
#     # - 32-bin histogram of Value (HSV)
#     # - 32-bin histogram of gradient magnitude (like LBP-ish)

#     try:
#         im = cv2.imread(p)  # Load image in BGR
#         if im is None:
#             return [0.0] * 128

#         # Resize for consistency
#         im = cv2.resize(im, (224, 224))

#         # HSV histograms
#         hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
#         h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
#         s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
#         v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()

#         # Gradient magnitude histogram
#         gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#         gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
#         gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
#         mag = np.sqrt(gx*gx + gy*gy)
#         lbp_hist, _ = np.histogram(mag.ravel(), bins=32, range=(0, 255))

#         # Concatenate to 128-D
#         feat = np.concatenate(
#             [h, s, v, lbp_hist.astype(np.float32)]).astype(np.float32)

#         # L2 normalization
#         nrm = np.linalg.norm(feat)
#         return (feat / nrm).tolist() if nrm > 1e-8 else [0.0]*128
#     except Exception:
#         # Any failure → return zero vector
#         return [0.0] * 128


# def _build_silhouette(hsv: np.ndarray) -> np.ndarray:
#     """
#     Build a fruit silhouette mask to exclude background but keep interior spots.
#     Steps:
#       1) Rough foreground by S/V thresholds (remove white/black background).
#       2) Morphological open/close to denoise.
#       3) Keep largest contour (assumed fruit body); dilate slightly.
#     Returns a uint8 mask {0,255}.
#     """
#     S = hsv[:, :, 1]
#     V = hsv[:, :, 2]

#     rough = ((S > 20) & (V > 15) & (V < 245)).astype(np.uint8) * 255
#     rough = cv2.morphologyEx(rough, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
#     rough = cv2.morphologyEx(rough, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

#     cnts, _ = cv2.findContours(rough, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     mask = np.zeros_like(rough)
#     if cnts:
#         c = max(cnts, key=cv2.contourArea)
#         cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
#         mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
#     else:
#         mask[:] = 255
#     return mask

# def featurize_path(p: str):
#     """
#     Load an image, compute 128-D features within the fruit silhouette:
#       - HSV histograms (H,S,V: 32 bins each)
#       - Gradient magnitude histogram (32 bins)
#     Return a Python list[float] length 128. On failure, return all zeros.
#     """
#     try:
#         im = cv2.imread(p)
#         if im is None:
#             return [0.0] * 128

#         # Standardize size
#         im = cv2.resize(im, (224, 224))

#         # HSV + mild lighting normalization on V (CLAHE)
#         hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
#         H, S, V = cv2.split(hsv)
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         V = clahe.apply(V)
#         hsv = cv2.merge([H, S, V])

#         # Build silhouette mask
#         mask = _build_silhouette(hsv)

#         # HSV histograms inside mask
#         h = cv2.calcHist([hsv], [0], mask, [32], [0, 180]).flatten()
#         s = cv2.calcHist([hsv], [1], mask, [32], [0, 256]).flatten()
#         v = cv2.calcHist([hsv], [2], mask, [32], [0, 256]).flatten()

#         # Gradient magnitude histogram inside mask
#         gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#         gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
#         gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
#         mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)

#         mag_vals = mag[mask.astype(bool)]
#         if mag_vals.size == 0:
#             grad_hist = np.zeros(32, dtype=np.float32)
#         else:
#             grad_hist, _ = np.histogram(mag_vals, bins=32, range=(0, 255))
#             grad_hist = grad_hist.astype(np.float32)

#         # Concatenate -> 128-D
#         feat = np.concatenate([h, s, v, grad_hist]).astype(np.float32)

#         # L2 normalize
#         nrm = np.linalg.norm(feat)
#         feat = feat / nrm if nrm > 1e-8 else np.zeros_like(feat)

#         return feat.tolist()
#     except Exception:
#         return [0.0] * 128
"""
    NEW extract features:
    128-D features focusing on the fruit object (contour-based mask, no hard color threshold):
      - resize to 224x224
      - crop constant borders (white/black) to reduce background influence
      - CLAHE on V channel (illumination normalization)
      - build mask from edges -> dilate/close -> largest convex-hull
      - histograms: HSV(24*3) + Lab a,b(16*2) + grad-mag(16)
      - stats: [H,S,V mean/std] + [edge density, dark ratio] = 8
      - per-block L1 then global L2
    Return: list[float] of length 128; fall back to zeros on failure.
    """
def featurize_path(p: str):
    
    try:
        im = cv2.imread(p)
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
    
# Register UDF in Spark
featurize_udf = F.udf(featurize_path, T.ArrayType(T.FloatType()))

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
