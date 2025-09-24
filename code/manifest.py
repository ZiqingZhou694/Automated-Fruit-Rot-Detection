# Purpose: Generate a manifest (CSV file) listing all image paths and their labels (fresh/rotten).
# The script scans the "data" folder recursively and saves the results to "out/manifest.csv".

import csv, pathlib, sys, shutil
# Root directory of your dataset 
# make sure "data/train" and "data/test" exist under this folder
ROOT = pathlib.Path("data").resolve()  
# Supported image file extensions
EXTS = {".jpg", ".jpeg", ".png"}

def infer_label(p: pathlib.Path):
    parts = [s.lower() for s in p.parts]
    for s in parts:
        if "fresh" in s:
            return "fresh"
        if "rotten" in s:
            return "rotten"
    return None

# Check if dataset exists
if not ROOT.exists():
    print(f"[ERR] data folder not found: {ROOT}")
    sys.exit(1)

rows = []

# Recursively scan all files under "data/"
for img in ROOT.rglob("*"):
    if img.is_file() and img.suffix.lower() in EXTS:
        label = infer_label(img.parent)
        if label:
            # 构造新的路径（把空格替换成下划线）
            safe_path = pathlib.Path(str(img).replace(" ", "_"))
            # 如果 safe_path 和原始文件不同，就复制一份
            if safe_path != img:
                safe_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img, safe_path)
            rows.append([str(safe_path.resolve()), label])

# Output directory (for manifest.csv)
out = pathlib.Path("out")
out.mkdir(exist_ok=True)

# Write CSV: two columns -> path, label
with open(out/"manifest.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f)
    w.writerow(["path","label"])
    w.writerows(rows)

print(f"[OK] wrote {out/'manifest.csv'} ({len(rows)} rows)")
