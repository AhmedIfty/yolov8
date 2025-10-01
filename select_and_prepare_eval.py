# save as tools/select_and_prepare_eval.py
import os, shutil, glob
from pathlib import Path

# EDIT THESE
DATASET_TEST_IMAGES = "bqa_evaluation/test-bqa/images"   # your real test images folder
DATASET_TEST_LABELS = "bqa_evaluation/test-bqa/labels"   # your real test labels folder
OUT_ROOT = "bqa-evaluation"
N_MAX = 50  # change to 30–50

CLASS_NAMES = ['glass','medical','metal','organic','paper','plastic','sharp-object']
MED_ID, SHARP_ID = 1, 6

def has_target_obj(label_path):
    if not os.path.exists(label_path):
        return False
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cid = int(parts[0])
            if cid in (MED_ID, SHARP_ID):
                return True
    return False

def main():
    out_img_dir = Path(OUT_ROOT) / "test-bqa" / "images"
    out_lbl_dir = Path(OUT_ROOT) / "test-bqa" / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    picked = 0
    for img_path in glob.glob(os.path.join(DATASET_TEST_IMAGES, "*")):
        stem = Path(img_path).stem
        lbl_path = os.path.join(DATASET_TEST_LABELS, f"{stem}.txt")
        if has_target_obj(lbl_path):
            shutil.copy2(img_path, out_img_dir / Path(img_path).name)
            if os.path.exists(lbl_path):
                shutil.copy2(lbl_path, out_lbl_dir / Path(lbl_path).name)
            picked += 1
            if picked >= N_MAX:
                break
    print(f"Copied {picked} images to {out_img_dir.parent}")

if __name__ == "__main__":
    main()
