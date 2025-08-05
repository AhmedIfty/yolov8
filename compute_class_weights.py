# compute_class_weights.py

import glob
from collections import Counter
from pathlib import Path

# 1️ Point this at your TRAIN labels folder
label_dir = Path('dataset/train/labels')

# 2️ Gather all .txt files
label_files = list(label_dir.glob('*.txt'))
if not label_files:
    raise FileNotFoundError(f"No label files found in {label_dir.resolve()}")

# 3️ Count how many times each class appears
counts = Counter()
for lf in label_files:
    for line in lf.read_text().splitlines():
        if line.strip():
            cls_id = int(line.split()[0])    # first token is class index
            counts[cls_id] += 1

# 4️ Arrange into a list (nc = number of classes)
nc = 7  # ← update if you have a different number of classes
class_counts = [counts[i] for i in range(nc)]
total = sum(class_counts)

# 5️ Compute inverse–frequency and scale so max≈5
inv_freq = [total/c if c>0 else 0.0 for c in class_counts]
scale = max(inv_freq) / 5
cls_weights = [round(w/scale, 2) for w in inv_freq]

# 6️ Print results
print("Class counts:             ", class_counts)
print("Inverse-frequency weights:", cls_weights)
