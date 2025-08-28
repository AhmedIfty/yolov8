import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
from ultralytics import YOLO

# ------------- utilities -------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def find_image_for_label(images_dir, label_fname_stem):
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
        p = os.path.join(images_dir, label_fname_stem + ext)
        if os.path.exists(p):
            return p
    return None

def load_yolo_txt(txt_path):
    boxes = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                # tolerate malformed lines like "class cx cy w h"
                # or extra spaces. Skip invalid lines
                try:
                    parts = [p for p in parts if p]
                    if len(parts) != 5:
                        continue
                except:
                    continue
            try:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
                boxes.append((cls, cx, cy, w, h))
            except:
                continue
    return boxes  # normalized xywh

def xywhn_to_xyxy_abs(cx, cy, w, h, W, H):
    x1 = (cx - w / 2.0) * W
    y1 = (cy - h / 2.0) * H
    x2 = (cx + w / 2.0) * W
    y2 = (cy + h / 2.0) * H
    return [x1, y1, x2, y2]

def clip_xyxy(x1, y1, x2, y2, W, H):
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    return [x1, y1, x2, y2]

def expand_with_context(x1, y1, x2, y2, W, H, context=0.15):
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w2 = w * (1.0 + 2.0 * context)
    h2 = h * (1.0 + 2.0 * context)
    x1n = cx - w2 * 0.5
    y1n = cy - h2 * 0.5
    x2n = cx + w2 * 0.5
    y2n = cy + h2 * 0.5
    return clip_xyxy(x1n, y1n, x2n, y2n, W, H)

def iou_xyxy(a, b):
    # a: [x1,y1,x2,y2], b: [x1,y1,x2,y2]
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    a_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    b_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = a_area + b_area - inter + 1e-9
    return inter / union

def preprocess_tensor(img_bgr, size=256, device="cpu"):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return t.to(device)

# ------------- embedding hook -------------

class DetectInputEmbedder:
    def __init__(self, yolo_model):
        self.model = yolo_model.model  # ultralytics.nn.tasks.DetectionModel
        self.device = next(self.model.parameters()).device
        self._buf = None
        # The Detect module is last in model.model
        self.detect = self.model.model[-1]
        self.hook = self.detect.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inputs, output):
        # inputs[0] is a list of feature maps [P3, P4, P5]
        feats_in = inputs[0]
        if isinstance(feats_in, (list, tuple)):
            pooled = [torch.nn.functional.adaptive_avg_pool2d(f, (1, 1)).flatten(1) for f in feats_in]
            concat = torch.cat(pooled, dim=1)  # [B, Csum]
        else:
            concat = torch.nn.functional.adaptive_avg_pool2d(feats_in, (1, 1)).flatten(1)
        self._buf = concat.detach().cpu().numpy()

    def embed(self, img_tensor):
        with torch.no_grad():
            _ = self.model(img_tensor)  # forward triggers hook
        if self._buf is None:
            raise RuntimeError("Hook did not capture features")
        vec = self._buf[0].copy()
        self._buf = None
        return vec

    def close(self):
        try:
            self.hook.remove()
        except:
            pass

# ------------- main pipeline -------------

def extract_from_gt(args):
    model = YOLO(args.model)
    model.model.eval().to(args.device)
    embedder = DetectInputEmbedder(model)

    X, y = [], []
    total_med, total_non = 0, 0

    label_files = [f for f in os.listdir(args.labels_dir) if f.endswith(".txt")]
    pbar = tqdm(label_files, desc="GT boxes")

    for lf in pbar:
        img_path = find_image_for_label(args.images_dir, os.path.splitext(lf)[0])
        if img_path is None:
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]

        boxes = load_yolo_txt(os.path.join(args.labels_dir, lf))
        if not boxes:
            continue

        for cls, cx, cy, w, h in boxes:
            x1, y1, x2, y2 = xywhn_to_xyxy_abs(cx, cy, w, h, W, H)
            x1, y1, x2, y2 = expand_with_context(x1, y1, x2, y2, W, H, context=args.context)
            bw = x2 - x1
            bh = y2 - y1
            if bw < args.min_box or bh < args.min_box:
                continue

            crop = img[int(y1):int(y2), int(x1):int(x2)]
            if crop.size == 0:
                continue

            t = preprocess_tensor(crop, size=args.input_size, device=args.device)
            vec = embedder.embed(t)
            X.append(vec)
            lbl = 1 if cls == args.medical_idx else 0
            y.append(lbl)
            if lbl == 1:
                total_med += 1
            else:
                total_non += 1

    embedder.close()

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    meta = dict(mode="gt", medical_idx=args.medical_idx,
                input_size=args.input_size, context=args.context,
                min_box=args.min_box, counts=dict(med=int(total_med), non=int(total_non)))

    os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
    np.savez(args.out, X=X, y=y, meta=meta)
    print(f"Saved {args.out}  X:{X.shape}  y:{y.shape}  counts:{meta['counts']}")

def extract_from_proposals(args):
    model = YOLO(args.model)
    model.model.eval().to(args.device)
    embedder = DetectInputEmbedder(model)

    X, y = [], []
    total_pos, total_neg = 0, 0

    label_files = [f for f in os.listdir(args.labels_dir) if f.endswith(".txt")]
    pbar = tqdm(label_files, desc="Proposals")

    for lf in pbar:
        stem = os.path.splitext(lf)[0]
        img_path = find_image_for_label(args.images_dir, stem)
        if img_path is None:
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]

        # load GT, keep only medical boxes for labeling
        boxes = load_yolo_txt(os.path.join(args.labels_dir, lf))
        gt_med = []
        for cls, cx, cy, w, h in boxes:
            if cls == args.medical_idx:
                gt_med.append(xywhn_to_xyxy_abs(cx, cy, w, h, W, H))

        # run model to get proposals
        with torch.no_grad():
            results = model.predict(source=img, conf=args.conf, iou=args.iou, device=args.device, verbose=False)
        if not results:
            continue
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue

        xyxy = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()

        # sort by confidence desc, take top-k if asked
        order = np.argsort(-scores)
        xyxy = xyxy[order]
        scores = scores[order]
        if args.topk > 0:
            xyxy = xyxy[:args.topk]
            scores = scores[:args.topk]

        for box in xyxy:
            x1, y1, x2, y2 = box.tolist()
            # expand a bit for context
            x1, y1, x2, y2 = expand_with_context(x1, y1, x2, y2, W, H, context=args.context)
            bw = x2 - x1
            bh = y2 - y1
            if bw < args.min_box or bh < args.min_box:
                continue

            # label by IoU with any medical GT
            lbl = 0
            for gm in gt_med:
                if iou_xyxy([x1, y1, x2, y2], gm) >= args.match_iou:
                    lbl = 1
                    break

            # optional negative subsampling
            if lbl == 0 and args.max_neg_pos_ratio > 0 and total_pos > 0:
                if total_neg >= args.max_neg_pos_ratio * total_pos:
                    continue

            crop = img[int(y1):int(y2), int(x1):int(x2)]
            if crop.size == 0:
                continue

            t = preprocess_tensor(crop, size=args.input_size, device=args.device)
            vec = embedder.embed(t)
            X.append(vec)
            y.append(lbl)
            if lbl == 1:
                total_pos += 1
            else:
                total_neg += 1

    embedder.close()

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    meta = dict(mode="proposals", medical_idx=args.medical_idx,
                input_size=args.input_size, context=args.context,
                min_box=args.min_box, conf=args.conf, iou=args.iou,
                match_iou=args.match_iou, topk=args.topk,
                max_neg_pos_ratio=args.max_neg_pos_ratio,
                counts=dict(pos=int(total_pos), neg=int(total_neg)))

    os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
    np.savez(args.out, X=X, y=y, meta=meta)
    print(f"Saved {args.out}  X:{X.shape}  y:{y.shape}  counts:{meta['counts']}")

def parse_args():
    ap = argparse.ArgumentParser("Extract YOLOv8 Detect-input embeddings for medical-vs-not proxy test")
    ap.add_argument("--model", type=str, required=True, help="Path to weights .pt")
    ap.add_argument("--images_dir", type=str, required=True)
    ap.add_argument("--labels_dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True, help="Output .npz path")
    ap.add_argument("--mode", type=str, choices=["gt", "proposals"], default="gt")

    ap.add_argument("--medical_idx", type=int, default=1, help="Index of 'medical' class")
    ap.add_argument("--input_size", type=int, default=256)
    ap.add_argument("--context", type=float, default=0.15, help="Relative expansion around box")
    ap.add_argument("--min_box", type=float, default=12.0, help="Discard boxes smaller than this in pixels")

    # proposals mode params
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--iou", type=float, default=0.60)
    ap.add_argument("--match_iou", type=float, default=0.50, help="IoU with medical GT for positive label")
    ap.add_argument("--topk", type=int, default=100, help="Top-K proposals per image, 0 for all")
    ap.add_argument("--max_neg_pos_ratio", type=float, default=3.0, help="Cap negatives at r times positives. 0 disables")

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.mode == "gt":
        extract_from_gt(args)
    else:
        extract_from_proposals(args)
