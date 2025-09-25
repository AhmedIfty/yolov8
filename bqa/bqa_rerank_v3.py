import argparse
from pathlib import Path
import re
import numpy as np
import cv2
from PIL import Image, ImageDraw
import torch
from ultralytics import YOLO
import open_clip
from tqdm import tqdm

# -----------------------------
# Defaults (same as your v1)
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMGZ = 640
YOLO_CONF = 0.15
YOLO_IOU  = 0.50
TOPK      = 75
LAMBDA    = 0.5
FUSE_THR  = 0.50

# === Dataset classes (order from your data.yaml) ===
CLASS_NAMES = ['glass', 'medical', 'metal', 'organic', 'paper', 'plastic', 'sharp-object']
NAME2ID = {n:i for i,n in enumerate(CLASS_NAMES)}

# ---------------------------------------------------
# Query routing config (edit/extend these in one place)
# Each group defines:
#   - keywords: substrings that map a free-form query to the group
#   - allow:    YOLO classes to KEEP for this query-group
#   - strict:   if True and gating yields zero boxes, return NO BOXES
#   - pos:      positive CLIP prompts (optional; defaults to raw query)
#   - neg:      negative CLIP prompts (optional; can be empty)
# ---------------------------------------------------
QUERY_GROUPS = {
    # ---- MEDICAL hazard queries -> gate to YOLO 'medical' ----
    "medical_hazard": {
        "keywords": [
            "syringe", "test tube", "vial", "lab tube", "mask", "face mask",
            "surgical mask", "glove", "gloves", "cotton", "bandage",
            "iv bag", "saline bag", "medicine packet", "drug packet", "blood tube"
        ],
        "allow": ["medical"],
        "strict": True,
        "pos": [],   # optional per-group positives (leave empty -> use user query)
        "neg": ["knife", "needle", "scissor", "axe", "blade", "screw", "metal rod"]
    },

    # ---- SHARP hazard queries -> gate to YOLO 'sharp-object' ----
    "sharp_hazard": {
        "keywords": [
            "scissor", "scissors", "knife", "needle", "pin", "axe", "ax",
            "fork", "blade", "screw", "box cutter", "cutter", "razor"
        ],
        "allow": ["sharp-object"],
        "strict": True,
        "pos": [],
        "neg": ["syringe", "test tube", "mask", "glove"]
    },

    # ---- Bottles (optional, nice demo) ----
    "plastic_bottle": {
        "keywords": ["plastic bottle", "water bottle", "drink bottle", "pet bottle"],
        "allow": ["plastic"],
        "strict": False,
        "pos": [],
        "neg": ["glass bottle", "wine bottle", "beer bottle", "glass"]
    },
    "glass_bottle": {
        "keywords": ["glass bottle", "beer bottle", "wine bottle"],
        "allow": ["glass"],
        "strict": False,
        "pos": [],
        "neg": ["plastic bottle", "mineral water bottle", "pet bottle", "plastic"]
    },
}

# ---- Helpers to normalize query and route to a group ----
_RX_WS = re.compile(r"\s+")
def _clean_query(q: str) -> str:
    # lower, collapse spaces, strip articles like "a picture of", "photo of", etc.
    q = q.lower().strip()
    q = q.replace("a picture of", "").replace("picture of", "").replace("photo of", "")
    q = q.replace("an image of", "").replace("image of", "")
    q = _RX_WS.sub(" ", q).strip()
    return q

def route_query(query: str):
    """Return a dict with:
        group_name, allow_ids (set[int]), strict (bool),
        pos_prompts (list[str]), neg_prompts (list[str])
       If no match: allow_ids=None (no gating), strict=False.
    """
    q = _clean_query(query)
    for gname, cfg in QUERY_GROUPS.items():
        for kw in cfg["keywords"]:
            if kw in q:
                allow_ids = {NAME2ID[n] for n in cfg["allow"] if n in NAME2ID}
                pos_prompts = cfg.get("pos", []) or [q]     # default to the cleaned query
                neg_prompts = cfg.get("neg", []) or []
                return {
                    "group_name": gname,
                    "allow_ids": allow_ids,
                    "strict": bool(cfg.get("strict", False)),
                    "pos_prompts": pos_prompts,
                    "neg_prompts": neg_prompts
                }
    # fallback: no gating
    return {
        "group_name": None,
        "allow_ids": None,
        "strict": False,
        "pos_prompts": [q],
        "neg_prompts": []
    }

# ---- CLIP model / scoring ----
def load_models(yolo_weights: str,
                clip_name: str = "ViT-B-32",
                clip_pretrained: str = "laion2b_s34b_b79k"):
    det = YOLO(yolo_weights)
    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_name, pretrained=clip_pretrained, device=DEVICE
    )
    tokenizer = open_clip.get_tokenizer(clip_name)
    model.eval()
    return det, model, preprocess, tokenizer

@torch.no_grad()
def clip_scores_for_boxes(clip_model, preprocess, tokenizer,
                          pos_prompts, neg_prompts,
                          image_bgr, boxes_xyxy, beta=0.0):
    """Score each crop against pos prompts; subtract beta * max(sim to neg) if neg provided."""
    # text features
    tok_pos = tokenizer(pos_prompts).to(DEVICE)
    txt_pos = clip_model.encode_text(tok_pos)
    txt_pos = txt_pos / txt_pos.norm(dim=-1, keepdim=True)

    if neg_prompts:
        tok_neg = tokenizer(neg_prompts).to(DEVICE)
        txt_neg = clip_model.encode_text(tok_neg)
        txt_neg = txt_neg / txt_neg.norm(dim=-1, keepdim=True)
    else:
        txt_neg = None

    H, W = image_bgr.shape[:2]
    crops, valid_idx = [], []
    for i, (x1,y1,x2,y2) in enumerate(boxes_xyxy):
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(W-1, int(x2)); y2 = min(H-1, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image_bgr[y1:y2, x1:x2, ::-1]  # BGR->RGB
        crops.append(Image.fromarray(crop))
        valid_idx.append(i)

    out = np.zeros((len(boxes_xyxy),), dtype=np.float32)
    if len(crops) == 0:
        return out

    batch = torch.cat([preprocess(c).unsqueeze(0) for c in crops], dim=0).to(DEVICE)
    img = clip_model.encode_image(batch)
    img = img / img.norm(dim=-1, keepdim=True)

    sims_pos = img @ txt_pos.T
    s = sims_pos.max(dim=1).values

    if txt_neg is not None and beta > 0.0:
        sims_neg = img @ txt_neg.T
        s = s - beta * sims_neg.max(dim=1).values

    for j, idx in enumerate(valid_idx):
        out[idx] = float(s[j].item())
    return out

# ---- Utilities ----
def norm01(x: np.ndarray):
    if x.size == 0:
        return x
    mn, mx = x.min(), x.max()
    if mx <= mn + 1e-9:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn + 1e-9)

def draw_boxes(image_bgr, boxes, scores, label="query", color=(0, 200, 255)):
    im = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(im)
    for (x1,y1,x2,y2), sc in zip(boxes, scores):
        dr.rectangle([x1,y1,x2,y2], outline=color, width=3)
        dr.text((x1+3, y1+3), f"{label} {sc:.2f}", fill=color)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

# ---- Main per-image routine ----
def run_on_image(det, clip_model, preprocess, tokenizer,
                 img_path: Path, query: str,
                 out_dir: Path = None,
                 topk: int = TOPK, fuse_lambda: float = LAMBDA, fuse_thr: float = FUSE_THR,
                 yolo_conf: float = YOLO_CONF, yolo_iou: float = YOLO_IOU, imgsz: int = IMGZ,
                 neg_beta: float = 0.0):
    im_bgr = cv2.imread(str(img_path))
    if im_bgr is None:
        raise FileNotFoundError(img_path)

    # 1) YOLO proposals
    r = det.predict(source=str(img_path), imgsz=imgsz, conf=yolo_conf, iou=yolo_iou,
                    verbose=False, save=False)[0]
    if r.boxes is None or r.boxes.xyxy.numel() == 0:
        return _save_rec(img_path, query, out_dir, im_bgr, [], [])

    boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
    yolo_scores = r.boxes.conf.cpu().numpy().astype(np.float32)
    yolo_cls = r.boxes.cls.cpu().numpy().astype(int)

    # 2) Route query -> allowed YOLO classes (gate)
    route = route_query(query)
    allow_ids = route["allow_ids"]  # None => no gating
    strict = route["strict"]
    pos_prompts = route["pos_prompts"]
    neg_prompts = route["neg_prompts"]

    if allow_ids is not None:
        gate = np.isin(yolo_cls, list(allow_ids))
        if gate.any():
            boxes = boxes[gate]
            yolo_scores = yolo_scores[gate]
            yolo_cls = yolo_cls[gate]
        elif strict:
            # strict hazard query but no allowed YOLO class present -> return no boxes
            return _save_rec(img_path, query, out_dir, im_bgr, [], [])

    if boxes.shape[0] == 0:
        return _save_rec(img_path, query, out_dir, im_bgr, [], [])

    # 3) keep top-k by YOLO score to control CLIP compute
    idx = np.argsort(-yolo_scores)[:topk]
    boxes = boxes[idx]; yolo_scores = yolo_scores[idx]

    # 4) CLIP scores (pos/neg). For now neg_beta default 0.0 as you requested to focus on gating.
    s_clip = clip_scores_for_boxes(clip_model, preprocess, tokenizer,
                                   pos_prompts, neg_prompts,
                                   im_bgr, boxes, beta=neg_beta)

    # 5) Fuse CLIP + YOLO and filter
    s_clip_n = norm01(s_clip)
    s_yolo_n = norm01(yolo_scores)
    fused = fuse_lambda * s_clip_n + (1.0 - fuse_lambda) * s_yolo_n

    keep = fused >= fuse_thr
    boxes_out = boxes[keep]
    scores_out = fused[keep]
    order = np.argsort(-scores_out)
    boxes_out = boxes_out[order]; scores_out = scores_out[order]

    return _save_rec(img_path, query, out_dir, im_bgr, boxes_out, scores_out)

def _save_rec(img_path, query, out_dir, im_bgr, boxes_out, scores_out):
    if len(boxes_out) > 0:
        vis = draw_boxes(im_bgr, boxes_out, scores_out, label=query)
    else:
        vis = im_bgr
    rec = {
        "image": str(img_path),
        "n_kept": int(len(boxes_out)),
        "boxes": (boxes_out.tolist() if hasattr(boxes_out, "tolist") else []),
        "scores": (scores_out.tolist() if hasattr(scores_out, "tolist") else [])
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_img = out_dir / f"{img_path.stem}_{_clean_query(query).replace(' ','_')}.jpg"
        cv2.imwrite(str(out_img), vis)
    return rec

# ---- CLI ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to YOLO baseline best.pt")
    ap.add_argument("--input", default="bqa/bqa_test", help="image file or folder")
    ap.add_argument("--query", required=True, help="free-form text (e.g., 'a picture of a syringe')")
    ap.add_argument("--output", default="bqa/bqa_result", help="folder to save visualizations")
    ap.add_argument("--clip", default="ViT-B-32")
    ap.add_argument("--clip_pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--fuse_lambda", type=float, default=LAMBDA)
    ap.add_argument("--fuse_thr", type=float, default=FUSE_THR)
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--yolo_conf", type=float, default=YOLO_CONF)
    ap.add_argument("--yolo_iou", type=float, default=YOLO_IOU)
    ap.add_argument("--imgsz", type=int, default=IMGZ)
    ap.add_argument("--neg_beta", type=float, default=0.0, help="0 = no negative prompts subtraction")
    args = ap.parse_args()

    det, clip_model, preprocess, tokenizer = load_models(
        yolo_weights=args.weights, clip_name=args.clip, clip_pretrained=args.clip_pretrained
    )

    in_path = Path(args.input)
    out_dir = Path(args.output)

    if in_path.is_file():
        rec = run_on_image(det, clip_model, preprocess, tokenizer, in_path, args.query,
                           out_dir, args.topk, args.fuse_lambda, args.fuse_thr,
                           args.yolo_conf, args.yolo_iou, args.imgsz, args.neg_beta)
        print(rec)
    else:
        imgs = sorted(list(in_path.glob("*.jpg")) + list(in_path.glob("*.png")) + list(in_path.glob("*.jpeg")))
        print(f"[INFO] Found {len(imgs)} images in {in_path}")
        out_csv = out_dir / f"bqa_{_clean_query(args.query).replace(' ','_')}.csv"
        rows = []
        for p in tqdm(imgs, desc="BQA"):
            rec = run_on_image(det, clip_model, preprocess, tokenizer, p, args.query,
                               out_dir, args.topk, args.fuse_lambda, args.fuse_thr,
                               args.yolo_conf, args.yolo_iou, args.imgsz, args.neg_beta)
            rows.append(rec)
        try:
            import pandas as pd
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"[OK] Wrote {out_csv}")
        except Exception as e:
            print("[WARN] Could not write CSV:", e)

if __name__ == "__main__":
    main()

# python bqa/bqa_rerank_v3.py --weights weights/yolov8m_refined_v4_baseline.pt --input path/to/image.jpg --query "a picture of a syringe"
# python bqa/bqa_rerank_v3.py --weights weights/yolov8m_refined_v4_baseline.pt --query "syringe"