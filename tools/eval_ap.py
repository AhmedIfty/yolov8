# tools/eval_ap.py
# Evaluate YOLO+CLIP predictions against gt_subtypes.csv
# Computes AP@0.5 per query (subtype) and macro average.
import os, csv, glob, argparse, math
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

CANON = {
    # sharp
    "knife":"knife","chef knife":"knife","kitchen knife":"knife","butter knife":"knife",
    "scissor":"scissors","scissors":"scissors","shears":"scissors",
    "fork":"fork",
    "razor":"razor","razor blade":"razor","blade":"razor","safety razor":"razor","double edged razor":"razor",
    "nail":"nail","metal nail":"nail","construction nail":"nail",
    "pin":"pin","thumbtack":"pin","push pin":"pin","tack":"pin","safety pin":"pin",
    # medical
    "syringe":"syringe",
    "test tube":"test tube","testtube":"test tube","blood tube":"test tube","vacutainer":"test tube",
    "mask":"mask","medical mask":"mask","surgical mask":"mask",
    "glove":"glove","medical glove":"glove",
}
def canon(s: str) -> str:
    s = (s or "").strip().lower()
    return CANON.get(s, s)

def iou_xyxy(a, b):
    # a,b = [x1,y1,x2,y2]
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def load_gt(gt_csv):
    """Return dict: gt_by_query[query] -> list of (image, [x1,y1,x2,y2])"""
    gt_by_query = defaultdict(list)
    with open(gt_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            sub = canon(row.get("subtype",""))
            if not sub:
                continue  # ignore unlabeled
            img = row["image"]
            x1,y1,x2,y2 = map(float, [row["x1"],row["y1"],row["x2"],row["y2"]])
            gt_by_query[sub].append((img, [x1,y1,x2,y2]))
    return gt_by_query

def load_preds(pred_paths):
    """Read multiple prediction CSVs -> dict: preds_by_query[query] = list of dicts"""
    preds_by_query = defaultdict(list)
    for p in pred_paths:
        with open(p, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                q = canon(row.get("query") or row.get("pred_subtype",""))
                if not q:
                    continue
                preds_by_query[q].append({
                    "image": row["image"],
                    "box": list(map(float, [row["x1"],row["y1"],row["x2"],row["y2"]])),
                    "score": float(row["score"]),
                    "pred_subtype": canon(row.get("pred_subtype","")),
                })
    # sort each list by score desc
    for q in preds_by_query:
        preds_by_query[q].sort(key=lambda d: d["score"], reverse=True)
    return preds_by_query

def match_and_score(preds, gts, iou_thr=0.5):
    """Greedy match preds to GT within each image; return arrays TP, FP, and GT count."""
    # Build GT availability per image
    gt_by_img = defaultdict(list)
    for (img, box) in gts:
        gt_by_img[img].append({"box": box, "matched": False})

    tp, fp = [], []
    for p in preds:
        img = p["image"]
        box = p["box"]
        if img not in gt_by_img or len(gt_by_img[img]) == 0:
            # no GT of this subtype in this image
            tp.append(0); fp.append(1)
            continue
        # find best IoU over unmatched GTs
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gt_by_img[img]):
            if g["matched"]:
                continue
            i = iou_xyxy(box, g["box"])
            if i > best_iou:
                best_iou, best_j = i, j
        if best_iou >= iou_thr:
            gt_by_img[img][best_j]["matched"] = True
            tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)

    # count GTs
    n_gt = sum(len(v) for v in gt_by_img.values())
    return np.array(tp, dtype=np.int32), np.array(fp, dtype=np.int32), int(n_gt)

def pr_from_tp_fp(tp, fp, n_gt):
    if n_gt == 0:
        return np.array([0.0]), np.array([1.0]), 0.0  # undefined; handled outside
    ctp = np.cumsum(tp)
    cfp = np.cumsum(fp)
    recall = ctp / (n_gt + 1e-9)
    precision = ctp / np.maximum(1, ctp + cfp)
    # AP: 11-point interpolated or numeric integration; use the standard monotonic precision envelope
    mpre = np.concatenate(([0.0], precision, [0.0]))
    mrec = np.concatenate(([0.0], recall,    [1.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    # integrate
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1]))
    return precision, recall, ap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="Path to bqa-evaluation/gt_subtypes.csv")
    ap.add_argument("--pred_glob", required=False, default="bqa_result/**/pred_*.csv",
                    help="Glob for prediction CSVs (one per query)")
    ap.add_argument("--pred_list", nargs="*", help="Explicit list of prediction CSVs (overrides glob)")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default="bqa-evaluation/eval_out")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load GT
    gt_by_query = load_gt(args.gt)
    gt_queries = sorted(gt_by_query.keys())
    if not gt_queries:
        print("[ERR] No labeled subtypes found in GT. Check gt_subtypes.csv")
        return

    # 2) load predictions
    if args.pred_list:
        pred_paths = args.pred_list
    else:
        pred_paths = glob.glob(args.pred_glob, recursive=True)
    if not pred_paths:
        print("[ERR] No prediction CSVs found with", args.pred_glob)
        return
    preds_by_query = load_preds(pred_paths)

    # 3) evaluate per query
    results = []
    macro_aps = []
    for q in gt_queries:
        preds = preds_by_query.get(q, [])
        gts   = gt_by_query[q]
        tp, fp, n_gt = match_and_score(preds, gts, iou_thr=args.iou)
        if n_gt == 0:
            results.append((q, 0, 0.0, 0, 0))
            continue
        prec, rec, ap = pr_from_tp_fp(tp, fp, n_gt)
        fn = n_gt - int(tp.sum())
        # Optional: best F1 over the sampled PR points
        f1 = 0.0
        if len(prec) and len(rec):
            f1 = float(np.max((2 * prec * rec) / np.maximum(prec + rec, 1e-9)))
        macro_aps.append(ap)
        # save PR curve
        pr_csv = out_dir / f"pr_{q.replace(' ','_')}.csv"
        with open(pr_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["precision","recall"])
            for P,R in zip(prec, rec): w.writerow([f"{P:.6f}", f"{R:.6f}"])
        # results.append((q, n_gt, ap, int(tp.sum()), int(fp.sum())))
        results.append((q, n_gt, ap, int(tp.sum()), int(fp.sum()), fn, f1))

    # 4) write summary table
    # sum_csv = out_dir / "summary.csv"
    # with open(sum_csv, "w", newline="", encoding="utf-8") as f:
    #     w = csv.writer(f)
    #     w.writerow(["query", "n_gt", "AP@0.5", "TP", "FP", "FN", "best_F1"])
    #     for q, n_gt, ap, tp_sum, fp_sum, fn, f1 in results:
    #         w.writerow([q, n_gt, f"{ap:.4f}", tp_sum, fp_sum, fn, f"{f1:.4f}"])
    #     if macro_aps:
    #         w.writerow([])
    #         w.writerow(["macro_avg", sum(n_gt for _,n_gt,_,_,_ in results), f"{np.mean(macro_aps):.4f}", "", ""])
    # print(f"[OK] Wrote {sum_csv.resolve()}")
    # ... after building `results` and `macro_aps` ...

    sum_csv = out_dir / "summary.csv"
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query", "n_gt", "AP@0.5", "TP", "FP", "FN", "best_F1"])
        for row in results:
            # row = (q, n_gt, ap, tp_sum, fp_sum, fn, f1)
            w.writerow([row[0], row[1], f"{row[2]:.4f}", row[3], row[4], row[5], f"{row[6]:.4f}"])

        if macro_aps:
            # total n_gt = sum of the 2nd element of each row
            total_n_gt = sum(r[1] for r in results)
            w.writerow([])
            w.writerow(["macro_avg", total_n_gt, f"{np.mean(macro_aps):.4f}", "", "", "", ""])

if __name__ == "__main__":
    main()

# python tools/eval_ap.py --gt bqa-evaluation/gt_subtypes.csv --pred_glob "bqa_result/syringe/pred_syringe.csv" --iou 0.3 --out bqa-evaluation/eval_out
# python tools/eval_ap.py --gt bqa-evaluation/gt_subtypes.csv --pred_list bqa_result/knife/pred_knife.csv bqa_result/nail/pred_nail.csv bqa_result/syringe/pred_syringe.csv 'bqa_result/test tube/pred_test tube.csv' --iou 0.3 --out bqa-evaluation/eval_out
# python tools/eval_ap.py --gt bqa-evaluation/gt_subtypes.csv --pred_list bqa_result/knife/pred_knife.csv bqa_result/nail/pred_nail.csv bqa_result/syringe/pred_syringe.csv 'bqa_result/test tube/pred_test tube.csv' --iou 0.5 --out bqa-evaluation/eval_out
# python tools\eval_ap.py --gt bqa-evaluation\gt_subtypes.csv --pred_list bqa_result\owlvit\pred_knife.csv bqa_result\owlvit\pred_nail.csv bqa_result\owlvit\pred_syringe.csv bqa_result\owlvit\pred_test_tube.csv --iou 0.5 --out bqa-evaluation\eval_out_owlvit
