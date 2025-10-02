# tools/eval_owlvit.py
import os, glob, csv, argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# ---- utils ----
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)

def nms(boxes, scores, iou_thr=0.5):
    idxs = np.argsort(-scores)
    keep = []
    while len(idxs):
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        ious = np.array([iou_xyxy(boxes[i], boxes[j]) for j in rest])
        idxs = rest[ious < iou_thr]
    return keep

# canonicalize to match GT labels
def canon(s):
    s = (s or "").strip().lower()
    MAP = {"scissor":"scissors", "razor blade":"razor", "blade":"razor"}
    return MAP.get(s, s)

def parse_queries(qstr):
    # "knife|kitchen knife|chef knife" -> ["knife","kitchen knife","chef knife"]
    return [s.strip() for s in (qstr or "").split("|") if s.strip()]

# ---- main runner ----
def run_owlvit_on_dir(img_dir, out_csv, query, model_id="google/owlvit-base-patch32",
                      score_thr=0.25, nms_iou=0.5, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = OwlViTProcessor.from_pretrained(model_id)
    model = OwlViTForObjectDetection.from_pretrained(model_id).to(device).eval()

    texts = parse_queries(query)
    if not texts:
        raise ValueError("Empty --query")
    canon_query = canon(texts[0])  # canonical label = first text
    print(f"[OWL-ViT] queries={texts} -> canon='{canon_query}'")

    images = []
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
        images += glob.glob(os.path.join(img_dir, ext))
    images = sorted(images)
    print(f"[OWL-ViT] images={len(images)}")

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image","query","x1","y1","x2","y2","score","pred_subtype"])

        for p in images:
            im = Image.open(p).convert("RGB")
            # multiple synonyms for a single image: wrap in list-of-list
            inputs = processor(text=[texts], images=im, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            target_sizes = torch.tensor([im.size[::-1]]).to(device)  # (H,W)
            results = processor.post_process(outputs=outputs, target_sizes=target_sizes)[0]
            boxes  = results["boxes"].cpu().numpy()   # [N,4] xyxy
            scores = results["scores"].cpu().numpy()  # [N]

            # score filter
            keep = np.where(scores >= score_thr)[0]
            boxes, scores = boxes[keep], scores[keep]

            # NMS
            if len(boxes) > 0 and nms_iou is not None:
                k = nms(boxes, scores, iou_thr=nms_iou)
                boxes, scores = boxes[k], scores[k]

            for b, s in zip(boxes, scores):
                x1,y1,x2,y2 = [float(v) for v in b.tolist()]
                w.writerow([os.path.basename(p), canon_query, x1,y1,x2,y2, float(s), canon_query])

    print(f"[OK] wrote {out_csv}")

# ---- CLI ----
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--query", required=True, help="Use '|' to add synonyms, e.g. 'knife|kitchen knife|chef knife'")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--score_thr", type=float, default=0.25)
    ap.add_argument("--nms_iou", type=float, default=0.5)
    ap.add_argument("--model", default="google/owlvit-base-patch32")
    args = ap.parse_args()

    run_owlvit_on_dir(args.input, args.out_csv, args.query,
                      model_id=args.model, score_thr=args.score_thr, nms_iou=args.nms_iou)


# python tools\eval_owlvit.py --input bqa-evaluation\test-bqa\images --query "knife|kitchen knife|chef knife|butter knife" --out_csv bqa_result\owlvit\pred_knife.csv --score_thr 0.05 --nms_iou 0.7 --model google/owlvit-base-patch16