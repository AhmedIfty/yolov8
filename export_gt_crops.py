# export_gt_crops.py
# ---------------------------------------------------
# Export GT crops for selected YOLO super-classes and
# create a CSV for manual sub-class annotation.
# - Saves both filenames AND full paths
# - Includes padded GT coords x1,y1,x2,y2 (pixels)
# - Debug prints for path resolution
# ---------------------------------------------------

import argparse, csv, os
from pathlib import Path
import yaml
import cv2

# Default super-classes we care about for Day-1
SHARP = "sharp-object"
MED   = "medical"

def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def yolo_txt_to_boxes(txt_path: Path):
    """
    Read YOLO .txt labels (class cx cy w h) with normalized coords.
    Returns: list of tuples (cls, cx, cy, w, h)
    """
    boxes = []
    if not txt_path.exists():
        return boxes
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            boxes.append((cls, cx, cy, w, h))
    return boxes

def xywhn_to_xyxy_pixels(cx, cy, w, h, W, H):
    """
    Convert normalized YOLO (cx,cy,w,h) to pixel xyxy.
    """
    x1 = (cx - w/2.0) * W
    y1 = (cy - h/2.0) * H
    x2 = (cx + w/2.0) * W
    y2 = (cy + h/2.0) * H
    return [x1, y1, x2, y2]

def pad_and_clamp(x1, y1, x2, y2, pad_ratio, W, H):
    """
    Pad by ratio and clamp to image bounds. Returns ints.
    """
    pw = (x2 - x1) * pad_ratio
    ph = (y2 - y1) * pad_ratio
    x1 -= pw; y1 -= ph; x2 += pw; y2 += ph
    x1 = max(0, min(int(x1), W-1))
    y1 = max(0, min(int(y1), H-1))
    x2 = max(0, min(int(x2), W))
    y2 = max(0, min(int(y2), H))
    if x2 <= x1: x2 = min(W, x1+1)
    if y2 <= y1: y2 = min(H, y1+1)
    return x1, y1, x2, y2

def find_images(dirpath: Path):
    hits = []
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
        hits += list(dirpath.rglob(ext))
    return hits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="data.yaml path")
    ap.add_argument("--split", type=str, default="test", choices=["test","val","valid","train"])
    ap.add_argument("--out", type=str, default="bqa_crops_day1")
    ap.add_argument("--pad", type=float, default=0.12, help="padding ratio around GT boxes")
    ap.add_argument("--only_super", type=str, default="medical,sharp-object",
                    help="Comma-separated list of super-classes to export (e.g., 'medical', 'sharp-object')")
    args = ap.parse_args()

    keep_supers = set([s.strip() for s in args.only_super.split(",") if s.strip()])

    data_yml = Path(args.data).resolve()
    cfg = load_yaml(data_yml)
    names = cfg.get("names", [])
    if isinstance(names, dict):
        # handle dict mapping style {0:'glass',1:'medical',...}
        names = [names[i] for i in sorted(names.keys(), key=lambda x: int(x))]

    base = data_yml.parent
    split_key = {"valid": "val"}.get(args.split, args.split)

    # Resolve split path from yaml
    if split_key == "test":
        img_rel = cfg["test"]
    elif split_key == "val":
        img_rel = cfg.get("val", cfg.get("valid"))
    else:
        img_rel = cfg[split_key]  # 'train' or 'val'

    def resolve_with_fallback(rel: str) -> Path:
        # primary: base / rel (ultralytics style)
        p = (base / rel).resolve()
        if find_images(p):
            return p
        # fallback: strip leading "../" components
        rel2 = rel.replace("../", "")
        p2 = (base / rel2).resolve()
        if find_images(p2):
            print(f"[INFO] Fallback path used for images: {p2}")
            return p2
        return p

    images_dir = resolve_with_fallback(img_rel)
    # guess labels dir by replacing /images with /labels
    labels_dir_guess = Path(str(images_dir).replace(os.sep+"images", os.sep+"labels"))
    if not labels_dir_guess.exists():
        # fallback label dir from yaml rel
        if "images" in img_rel:
            lbl_rel = img_rel.replace("images", "labels")
        else:
            lbl_rel = img_rel
        labels_dir_guess = resolve_with_fallback(lbl_rel)

    print(f"[DEBUG] data.yaml       : {data_yml}")
    print(f"[DEBUG] images_dir (raw): {img_rel}")
    print(f"[DEBUG] images_dir (res): {images_dir}")
    print(f"[DEBUG] labels_dir (res): {labels_dir_guess}")

    out_root = Path(args.out).resolve()
    crops_root = out_root / "crops"
    crops_root.mkdir(parents=True, exist_ok=True)

    csv_path = out_root / "annotated_crops_day1.csv"

    img_paths = find_images(images_dir)
    if not img_paths:
        print("[WARN] No images found. Check the [DEBUG] paths above, or update data.yaml to use: test: test/images")
        return

    rows = []
    kept = 0
    missing_lbl = 0
    no_keep = 0

    for p in sorted(img_paths):
        rel = p.relative_to(images_dir)
        lbl = (labels_dir_guess / rel).with_suffix(".txt")
        if not lbl.exists():
            missing_lbl += 1
            continue

        im = cv2.imread(str(p))
        if im is None:
            continue
        H, W = im.shape[:2]
        boxes = yolo_txt_to_boxes(lbl)

        for bi, (cls, cx, cy, w, h) in enumerate(boxes):
            if cls < 0 or cls >= len(names):
                continue
            super_name = names[cls]
            if super_name not in keep_supers:
                no_keep += 1
                continue

            # Compute padded GT coords and crop
            x1, y1, x2, y2 = xywhn_to_xyxy_pixels(cx, cy, w, h, W, H)
            x1, y1, x2, y2 = pad_and_clamp(x1, y1, x2, y2, args.pad, W, H)
            crop = im[y1:y2, x1:x2, :]
            if crop.size == 0:
                continue

            # Save crop under crops/<super_name>/
            out_dir = crops_root / super_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"{p.stem}_{bi:03d}.jpg"
            out_path = (out_dir / out_name).resolve()
            cv2.imwrite(str(out_path), crop)

            # Store both names and full paths + coords
            rows.append({
                "img_crop_name":   out_name,          # filename only
                "img_crop_path":   str(out_path),     # full path
                "orig_image_name": p.name,            # filename only
                "orig_image_path": str(p),            # full path
                "superclass":      super_name,        # e.g., 'medical' or 'sharp-object'
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,   # padded GT box (pixels)
                "subclass": "",                        # you will annotate
            })
            kept += 1

    # Write CSV (header must match row keys)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "img_crop_name", "img_crop_path",
            "orig_image_name", "orig_image_path",
            "superclass", "x1", "y1", "x2", "y2",
            "subclass"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[OK] Exported {kept} crops to {crops_root}")
    print(f"[INFO] Missing label files: {missing_lbl}")
    print(f"[INFO] Non-target boxes skipped: {no_keep}")
    print(f"[NEXT] Annotate subclasses in: {csv_path}")

if __name__ == "__main__":
    main()
