# tools/make_gt_crops.py
# Generate helper crops from YOLO GT + write a GT template CSV (blank subtype).
# Now saves crops in BOTH:
#   - per-image folders:   bqa-evaluation/yolo-crops/<imgstem>/imgstem_crop_0000.jpg
#   - one flat folder:     bqa-evaluation/yolo-crops-all/imgstem_crop_0000.jpg
#
# CSV columns include both relative crop paths so you can use whichever is easier.

import os, csv, glob
from pathlib import Path
import cv2

# ====== PATHS (edit as needed) ======
ROOT     = "../bqa-evaluation"
IMG_DIR  = f"{ROOT}/test-bqa/images"
LBL_DIR  = f"{ROOT}/test-bqa/labels"
OUT_TREE = f"{ROOT}/yolo-crops"      # per-image folders
OUT_FLAT = f"{ROOT}/yolo-crops-all"  # single flat folder
CSV_OUT  = f"{ROOT}/gt_template.csv"

# ====== DATASET CLASS MAPPING (edit if your IDs differ) ======
CLASS_NAMES = ['glass','medical','metal','organic','paper','plastic','sharp-object']
MED_ID, SHARP_ID = 1, 6

# ====== I/O helpers ======
def yolo_xywhn_to_xyxy(cx, cy, w, h, W, H):
    """YOLO normalized cx,cy,w,h -> pixel x1,y1,x2,y2 (clamped)."""
    x1 = (cx - w/2.0) * W
    y1 = (cy - h/2.0) * H
    x2 = (cx + w/2.0) * W
    y2 = (cy + h/2.0) * H
    x1 = max(0, int(round(x1))); y1 = max(0, int(round(y1)))
    x2 = min(W-1, int(round(x2))); y2 = min(H-1, int(round(y2)))
    if x2 <= x1: x2 = min(W-1, x1 + 1)
    if y2 <= y1: y2 = min(H-1, y1 + 1)
    return x1, y1, x2, y2

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    ensure_dir(Path(OUT_TREE))
    ensure_dir(Path(OUT_FLAT))

    rows = []
    img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*")))
    total_written = 0

    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Skipped unreadable {img_path}")
            continue
        H, W = img.shape[:2]
        stem = Path(img_path).stem
        lbl_path = os.path.join(LBL_DIR, f"{stem}.txt")
        if not os.path.exists(lbl_path):
            # silently ignore images without labels in this small eval slice
            continue

        # read labels
        with open(lbl_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        # per-image crop folder
        out_dir_tree = Path(OUT_TREE) / stem
        ensure_dir(out_dir_tree)

        idx = 0
        for ln in lines:
            parts = ln.split()
            try:
                cid = int(parts[0])
            except:
                continue

            # only medical + sharp-object (extend if you want others)
            if cid not in (MED_ID, SHARP_ID):
                continue

            cx, cy, w, h = map(float, parts[1:5])
            x1, y1, x2, y2 = yolo_xywhn_to_xyxy(cx, cy, w, h, W, H)

            # crop from original image
            crop = img[y1:y2, x1:x2].copy()

            # filenames (both locations)
            crop_base = f"{stem}_crop_{idx:04d}.jpg"
            crop_tree_rel = f"{stem}/{crop_base}"
            crop_flat_rel = crop_base

            # write to tree folder
            tree_path = out_dir_tree / crop_base
            cv2.imwrite(str(tree_path), crop)

            # write to flat folder
            flat_path = Path(OUT_FLAT) / crop_base
            cv2.imwrite(str(flat_path), crop)

            # CSV row
            rows.append({
                "crop_name": crop_tree_rel,   # per-image folder path
                "crop_name_flat": crop_flat_rel,  # flat-folder path
                "image": Path(img_path).name,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "superclass": CLASS_NAMES[cid],
                "subtype": ""  # fill manually later
            })

            idx += 1
            total_written += 1

    # write CSV
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "crop_name","crop_name_flat","image",
                "x1","y1","x2","y2",
                "superclass","subtype"
            ]
        )
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] Wrote {len(rows)} rows to {CSV_OUT}")
    print(f"[OK] Crops written to:\n  - {OUT_TREE}\n  - {OUT_FLAT}\nTotal crops: {total_written}")

if __name__ == "__main__":
    main()
