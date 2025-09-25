import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from ultralytics import YOLO

# ---------- Config ----------
NAMES = ['glass','medical','metal','organic','paper','plastic','sharp-object']
CLS_TO_NAME = {i:n for i,n in enumerate(NAMES)}

# ---------- Geometry / IOU ----------
def box_iou_xyxy(a, b):
    # a: [Na,4], b: [Nb,4], xyxy
    a = a.astype(np.float32); b = b.astype(np.float32)
    Na, Nb = a.shape[0], b.shape[0]
    if Na==0 or Nb==0:
        return np.zeros((Na, Nb), dtype=np.float32)
    x1 = np.maximum(a[:,None,0], b[None,:,0])
    y1 = np.maximum(a[:,None,1], b[None,:,1])
    x2 = np.minimum(a[:,None,2], b[None,:,2])
    y2 = np.minimum(a[:,None,3], b[None,:,3])
    inter = np.clip(x2-x1, a_min=0, a_max=None) * np.clip(y2-y1, a_min=0, a_max=None)
    area_a = (a[:,2]-a[:,0]) * (a[:,3]-a[:,1])
    area_b = (b[:,2]-b[:,0]) * (b[:,3]-b[:,1])
    union = area_a[:,None] + area_b[None,:] - inter + 1e-9
    return inter / union

def greedy_match_per_class(pred_boxes, gt_boxes, iou_thr=0.5):
    """Return TP, FP, FN counts for one class using greedy IoU matching."""
    if len(pred_boxes)==0 and len(gt_boxes)==0:
        return 0, 0, 0
    if len(pred_boxes)==0:
        return 0, 0, len(gt_boxes)
    if len(gt_boxes)==0:
        return 0, len(pred_boxes), 0
    IoU = box_iou_xyxy(pred_boxes, gt_boxes)
    TP = 0
    used_pred = np.zeros(IoU.shape[0], dtype=bool)
    used_gt   = np.zeros(IoU.shape[1], dtype=bool)
    # Greedy: repeatedly pick highest IoU
    while True:
        i, j = np.unravel_index(np.argmax(IoU), IoU.shape)
        if IoU[i, j] < iou_thr:
            break
        if used_pred[i] or used_gt[j]:
            IoU[i, j] = -1.0
            continue
        TP += 1
        used_pred[i] = True
        used_gt[j] = True
        IoU[i, :] = -1.0
        IoU[:, j] = -1.0
    FP = int((~used_pred).sum())
    FN = int((~used_gt).sum())
    return TP, FP, FN

def tp_fp_fn_all_classes(pred_boxes, pred_cls, gt_boxes, gt_cls, iou_thr=0.5, num_classes=7):
    TP=FP=FN=0
    per_class = {}
    for c in range(num_classes):
        p = pred_boxes[pred_cls==c]
        g = gt_boxes[gt_cls==c]
        tpc, fpc, fnc = greedy_match_per_class(p, g, iou_thr=iou_thr)
        per_class[c] = (tpc, fpc, fnc)
        TP += tpc; FP += fpc; FN += fnc
    return TP, FP, FN, per_class

def f1_from_counts(tp, fp, fn):
    prec = tp / max(tp+fp, 1)
    rec  = tp / max(tp+fn, 1)
    return 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)

# ---------- Features ----------
def entropy_gray(gray):
    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).ravel()
    p = hist / (hist.sum()+1e-9)
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p*np.log(p)).sum())

def lap_var(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def orb_kp_density(img_bgr, scale=0.25, max_kp=800):
    h,w = img_bgr.shape[:2]
    sm = cv2.resize(img_bgr, (int(w*scale), int(h*scale)))
    gray = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(max_kp)
    kps = orb.detect(gray, None)
    return float(len(kps)) / max(1, (sm.shape[0]*sm.shape[1]))

def read_yolo_labels(txt_path, img_w, img_h):
    # Returns xyxy (pixels), cls (ints)
    if not txt_path.exists():
        return np.zeros((0,4), dtype=np.float32), np.zeros((0,), dtype=int)
    lines = [l.strip() for l in txt_path.read_text().splitlines() if l.strip()]
    boxes=[]; clses=[]
    for ln in lines:
        sp = ln.split()
        c = int(float(sp[0])); cx = float(sp[1]); cy = float(sp[2]); w = float(sp[3]); h = float(sp[4])
        # normalized cx,cy,w,h -> xyxy pixels
        bw = w * img_w; bh = h * img_h
        x1 = (cx * img_w) - bw/2; y1 = (cy * img_h) - bh/2
        x2 = x1 + bw; y2 = y1 + bh
        boxes.append([x1,y1,x2,y2]); clses.append(c)
    return np.array(boxes, dtype=np.float32), np.array(clses, dtype=int)

def small_obj_ratio_from_lbl(txt_path):
    if not txt_path.exists():
        return 0.0, {}
    lines = [l.strip() for l in txt_path.read_text().splitlines() if l.strip()]
    if not lines:
        return 0.0, {}
    areas=[]; cls_cnt={}
    for ln in lines:
        sp = ln.split()
        c = int(float(sp[0])); w = float(sp[3]); h = float(sp[4])
        cls_cnt[c] = cls_cnt.get(c,0)+1
        areas.append(w*h)  # normalized area
    areas = np.array(areas, dtype=np.float32)
    small = (areas < 0.02).sum()  # < 2% of image area
    return float(small)/len(areas), cls_cnt

# ---------- Main ----------
def run(args):
    val_imgs = Path(args.val_images)
    val_lbls = Path(args.val_labels)
    out_csv  = Path(args.out_csv)

    model_base = YOLO(args.baseline)
    model_p2   = YOLO(args.p2)

    rows = []
    img_paths = sorted(list(val_imgs.glob("*.jpg")) + list(val_imgs.glob("*.png")) + list(val_imgs.glob("*.jpeg")))
    for img_path in img_paths:
        im = cv2.imread(str(img_path))
        if im is None:
            print(f"[WARN] Cannot read {img_path}"); continue
        h,w = im.shape[:2]
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # features
        H  = entropy_gray(gray)
        LV = lap_var(gray)
        KD = orb_kp_density(im)

        # GT
        lbl_path = val_lbls / (img_path.stem + ".txt")
        gt_boxes, gt_cls = read_yolo_labels(lbl_path, w, h)
        small_ratio, cls_cnt = small_obj_ratio_from_lbl(lbl_path)

        # Predict baseline
        res_b = model_base.predict(source=str(img_path), imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False, save=False)
        rb = res_b[0]
        b_boxes = rb.boxes.xyxy.cpu().numpy().astype(np.float32) if rb.boxes is not None else np.zeros((0,4), dtype=np.float32)
        b_cls   = rb.boxes.cls.cpu().numpy().astype(int) if rb.boxes is not None else np.zeros((0,), dtype=int)

        # Predict P2
        res_p = model_p2.predict(source=str(img_path), imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False, save=False)
        rp = res_p[0]
        p_boxes = rp.boxes.xyxy.cpu().numpy().astype(np.float32) if rp.boxes is not None else np.zeros((0,4), dtype=np.float32)
        p_cls   = rp.boxes.cls.cpu().numpy().astype(int) if rp.boxes is not None else np.zeros((0,), dtype=int)

        # Metrics at fixed IoU
        TPb, FPb, FNb, per_b = tp_fp_fn_all_classes(b_boxes, b_cls, gt_boxes, gt_cls, iou_thr=args.iou_match, num_classes=len(NAMES))
        TPp, FPp, FNp, per_p = tp_fp_fn_all_classes(p_boxes, p_cls, gt_boxes, gt_cls, iou_thr=args.iou_match, num_classes=len(NAMES))

        F1b = f1_from_counts(TPb, FPb, FNb)
        F1p = f1_from_counts(TPp, FPp, FNp)
        dF1 = F1p - F1b

        rows.append({
            "img": img_path.name,
            "H": H, "lapvar": LV, "kpdens": KD, "small_ratio": small_ratio,
            "gt_count": int(gt_boxes.shape[0]),
            "gt_per_class": json.dumps(cls_cnt),
            "TP_base": TPb, "FP_base": FPb, "FN_base": FNb, "F1_base": F1b,
            "TP_p2":   TPp, "FP_p2":   FPp, "FN_p2":   FNp, "F1_p2":   F1p,
            "dF1": dF1
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote {out_csv} with {len(df)} rows.")
    print(df.describe(include='all'))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline",   default="weights/yolov8m_refined_v4_baseline.pt", help="path to baseline best.pt")
    ap.add_argument("--p2",         default="weights/yolov8m_refined_v4_p2.pt", help="path to P2 best.pt")
    ap.add_argument("--val_images", default="dataset-refined-v4/valid/images", help=".../valid/images")
    ap.add_argument("--val_labels", default="dataset-refined-v4/valid/labels", help=".../valid/labels")
    ap.add_argument("--out_csv",    default="val_compare.csv")
    ap.add_argument("--imgsz",      type=int, default=640)
    ap.add_argument("--conf",       type=float, default=0.25)
    ap.add_argument("--iou",        type=float, default=0.45, help="NMS IoU for prediction")
    ap.add_argument("--iou_match",  type=float, default=0.5,  help="IoU threshold for TP matching")
    args = ap.parse_args()
    run(args)
