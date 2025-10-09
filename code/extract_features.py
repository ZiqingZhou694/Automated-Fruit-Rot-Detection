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

from pyspark.sql import SparkSession, functions as F, types as T
import sys
import numpy as np
import cv2


def extract_features(p: str):
    
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
    