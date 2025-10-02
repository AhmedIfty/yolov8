# tools/vis_from_csv.py
import os, csv, argparse
from pathlib import Path
import cv2

def draw(img, boxes, labels, scores):
    for (x1,y1,x2,y2), lab, sc in zip(boxes, labels, scores):
        x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,255), 2)
        txt = f"{lab} {sc:.2f}"
        (tw,th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1- th - 6), (x1+tw+4, y1), (0,255,255), -1)
        cv2.putText(img, txt, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder with original images")
    ap.add_argument("--pred_csv", required=True, help="pred_*.csv from OWL-ViT or BQA")
    ap.add_argument("--out", required=True, help="output folder for visualizations")
    ap.add_argument("--score_thr", type=float, default=0.0, help="draw only boxes >= this score")
    ap.add_argument("--topk", type=int, default=0, help="keep top-k boxes per image after thr (0=all)")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    # load predictions grouped by image
    preds = {}
    with open(args.pred_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            img = row["image"]
            b = [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])]
            sc = float(row["score"])
            lb = row.get("pred_subtype") or row.get("query") or "obj"
            if sc < args.score_thr:
                continue
            preds.setdefault(img, []).append((b, lb, sc))

    # draw
    n_written = 0
    for img_name, items in preds.items():
        # sort by score, keep topk if set
        items.sort(key=lambda t: t[2], reverse=True)
        if args.topk > 0:
            items = items[:args.topk]

        img_path = Path(args.images) / img_name
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        boxes  = [it[0] for it in items]
        labels = [it[1] for it in items]
        scores = [it[2] for it in items]
        vis = draw(img, boxes, labels, scores)
        cv2.imwrite(str(Path(args.out) / img_name), vis)
        n_written += 1

    print(f"[OK] wrote {n_written} annotated images to {args.out}")

if __name__ == "__main__":
    main()

# python tools\vis_from_csv.py --images bqa-evaluation\test-bqa\images --pred_csv bqa_result\owlvit\pred_syringe.csv --out bqa_result\owlvit\vis_syringe --score_thr 0.05 --topk 0