# bqa_rerank.py

import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw
import torch
from ultralytics import YOLO
import open_clip
from tqdm import tqdm

# -----------------------------
# Defaults tuned for your setup
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMGZ = 640                  # proposal size: matches your train/val
YOLO_CONF = 0.15            # lower than val/test to collect more proposals
YOLO_IOU  = 0.50
TOPK      = 75              # limit CLIP scoring cost (8GB-friendly)
LAMBDA    = 0.5             # fuse weight: final = λ*CLIP + (1-λ)*YOLO
FUSE_THR  = 0.50            # filter after fusion (tune on val)

# Optional synonyms (extend freely)
QUERY_CANON = {
    # bottles
    "plastic bottle": ["plastic bottle", "water bottle", "drink bottle", "mineral water bottle"],
    "glass bottle":   ["glass bottle", "beer bottle", "wine bottle"],
    # medical
    "syringe": ["syringe", "injection syringe"],
    "test tube": ["test tube", "vial", "lab tube"],
    "mask": ["face mask", "surgical mask"],
    "gloves": ["disposable gloves", "medical gloves", "latex gloves"],
    # sharp
    "needle": ["needle", "hypodermic needle", "sewing needle"],
    "knife": ["knife", "box cutter", "utility knife", "scalpel"],
}

# === Your dataset classes (order from data.yaml) ===
CLASS_NAMES = ['glass', 'medical', 'metal', 'organic', 'paper', 'plastic', 'sharp-object']
NAME2ID = {n:i for i,n in enumerate(CLASS_NAMES)}

# === Map query -> which YOLO classes are allowed (gate) and negative prompts for CLIP ===
QUERY_ROUTING = {
    # hazards
    "syringe": {
        "allow": ["medical"],                                     # only medical boxes are valid candidates
        "neg":   ["knife", "needle", "test tube", "vial", "pipette", "metal rod", "scissors"]
    },
    "needle": {
        "allow": ["sharp-object"],
        "neg":   ["syringe", "test tube", "pipette", "wire", "nail", "metal rod"]
    },
    "knife": {
        "allow": ["sharp-object"],
        "neg":   ["syringe", "test tube", "scalpel handle", "metal rod"]
    },

    # bottles
    "plastic bottle": {
        "allow": ["plastic"],
        "neg":   ["glass bottle", "glass", "wine bottle", "beer bottle"]
    },
    "glass bottle": {
        "allow": ["glass"],
        "neg":   ["plastic bottle", "mineral water bottle", "PET bottle"]
    },

    # you can add "mask", "gloves", "test tube" etc., similarly
    "mask": {
        "allow": ["medical"],
        "neg":   ["glove", "cloth", "paper", "tissue"]
    },
    "test tube": {
        "allow": ["medical"],
        "neg":   ["syringe", "pipette", "bottle", "vial"]
    },
}


def load_models(yolo_weights: str,
                clip_name: str = "ViT-B-32",
                clip_pretrained: str = "laion2b_s34b_b79k"):
    # YOLO detector
    det = YOLO(yolo_weights)
    # CLIP
    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_name, pretrained=clip_pretrained, device=DEVICE
    )
    tokenizer = open_clip.get_tokenizer(clip_name)
    model.eval()
    return det, model, preprocess, tokenizer

@torch.no_grad()
def clip_scores_for_boxes(clip_model, preprocess, text_tokens, image_bgr, boxes_xyxy):
    """Return CLIP similarity for each box vs text (max over synonym prompts)."""
    # Encode text once
    txt = clip_model.encode_text(text_tokens.to(DEVICE))
    txt = txt / txt.norm(dim=-1, keepdim=True)

    H, W = image_bgr.shape[:2]
    crops = []
    valid_idx = []
    for i, (x1,y1,x2,y2) in enumerate(boxes_xyxy):
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(W-1, int(x2)); y2 = min(H-1, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image_bgr[y1:y2, x1:x2, ::-1]  # BGR->RGB
        crops.append(Image.fromarray(crop))
        valid_idx.append(i)
    if len(crops) == 0:
        return np.zeros((len(boxes_xyxy),), dtype=np.float32)

    batch = torch.cat([preprocess(c).unsqueeze(0) for c in crops], dim=0).to(DEVICE)
    img = clip_model.encode_image(batch)
    img = img / img.norm(dim=-1, keepdim=True)

    sims = (img @ txt.T)  # cosine (normalized)
    sims_max, _ = sims.max(dim=1)  # max over synonyms
    out = np.zeros((len(boxes_xyxy),), dtype=np.float32)
    for j, idx in enumerate(valid_idx):
        out[idx] = float(sims_max[j].item())
    return out

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

def run_on_image(det, clip_model, preprocess, tokenizer,
                 img_path: Path, query: str,
                 out_dir: Path = None,
                 topk: int = TOPK, fuse_lambda: float = LAMBDA, fuse_thr: float = FUSE_THR,
                 yolo_conf: float = YOLO_CONF, yolo_iou: float = YOLO_IOU, imgsz: int = IMGZ):
    im_bgr = cv2.imread(str(img_path))
    if im_bgr is None:
        raise FileNotFoundError(img_path)
    H, W = im_bgr.shape[:2]

    # 1) YOLO proposals (lower conf for recall)
    r = det.predict(source=str(img_path), imgsz=imgsz, conf=yolo_conf, iou=yolo_iou,
                    verbose=False, save=False)[0]
    if r.boxes is None or r.boxes.xyxy.numel() == 0:
        return {"image": str(img_path), "n_proposals": 0, "n_kept": 0}

    boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
    yolo_scores = r.boxes.conf.cpu().numpy().astype(np.float32)

    # keep top-k by YOLO score
    idx = np.argsort(-yolo_scores)[:topk]
    boxes = boxes[idx]; yolo_scores = yolo_scores[idx]

    # 2) CLIP text tokens (with synonyms if we have)
    synonyms = QUERY_CANON.get(query.lower(), [query])
    text_tokens = tokenizer(synonyms)

    # 3) CLIP scores for each box
    s_clip = clip_scores_for_boxes(clip_model, preprocess, text_tokens, im_bgr, boxes)

    # 4) Fuse CLIP + YOLO (normalize each to [0,1] first)
    s_clip_n = norm01(s_clip)
    s_yolo_n = norm01(yolo_scores)
    fused = fuse_lambda * s_clip_n + (1.0 - fuse_lambda) * s_yolo_n

    # 5) Filter & sort
    keep = fused >= fuse_thr
    boxes_out = boxes[keep]; scores_out = fused[keep]
    order = np.argsort(-scores_out)
    boxes_out = boxes_out[order]; scores_out = scores_out[order]

    vis = draw_boxes(im_bgr, boxes_out, scores_out, label=query)
    rec = {
        "image": str(img_path),
        "n_proposals": int(len(boxes)),
        "n_kept": int(len(boxes_out)),
        "boxes": boxes_out.tolist(),
        "scores": scores_out.tolist()
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_img = out_dir / f"{img_path.stem}_{query.replace(' ','_')}.jpg"
        cv2.imwrite(str(out_img), vis)
        # optional: save per-image json alongside
        # (commented to reduce clutter)
        # import json; json.dump(rec, open(out_dir / f"{img_path.stem}_{query}.json","w"), indent=2)

    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to YOLO baseline best.pt")
    ap.add_argument("--input", default="bqa/bqa_test", help="image file or folder")
    ap.add_argument("--query", required=True, help="e.g., 'syringe', 'plastic bottle'")
    ap.add_argument("--output", default="bqa/bqa_result", help="folder to save visualizations")
    ap.add_argument("--clip", default="ViT-B-32", help="open-clip model name")
    ap.add_argument("--clip_pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--fuse_lambda", type=float, default=LAMBDA)
    ap.add_argument("--fuse_thr", type=float, default=FUSE_THR)
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--yolo_conf", type=float, default=YOLO_CONF)
    ap.add_argument("--yolo_iou", type=float, default=YOLO_IOU)
    ap.add_argument("--imgsz", type=int, default=IMGZ)
    args = ap.parse_args()

    det, clip_model, preprocess, tokenizer = load_models(
        yolo_weights=args.weights, clip_name=args.clip, clip_pretrained=args.clip_pretrained
    )

    in_path = Path(args.input)
    out_dir = Path(args.output)

    if in_path.is_file():
        rec = run_on_image(det, clip_model, preprocess, open_clip.get_tokenizer(args.clip),
                           in_path, args.query, out_dir,
                           topk=args.topk, fuse_lambda=args.fuse_lambda, fuse_thr=args.fuse_thr,
                           yolo_conf=args.yolo_conf, yolo_iou=args.yolo_iou, imgsz=args.imgsz)
        print(rec)
    else:
        imgs = sorted(list(in_path.glob("*.jpg")) + list(in_path.glob("*.png")) + list(in_path.glob("*.jpeg")))
        print(f"[INFO] Found {len(imgs)} images in {in_path}")
        out_csv = out_dir / f"bqa_{args.query.replace(' ','_')}.csv"
        rows = []
        for p in tqdm(imgs, desc="BQA"):
            rec = run_on_image(det, clip_model, preprocess, open_clip.get_tokenizer(args.clip),
                               p, args.query, out_dir,
                               topk=args.topk, fuse_lambda=args.fuse_lambda, fuse_thr=args.fuse_thr,
                               yolo_conf=args.yolo_conf, yolo_iou=args.yolo_iou, imgsz=args.imgsz)
            rows.append(rec)
        # Save a simple CSV summary
        try:
            import pandas as pd
            df = pd.DataFrame(rows)
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv, index=False)
            print(f"[OK] Wrote {out_csv}")
        except Exception as e:
            print("[WARN] Could not write CSV:", e)

if __name__ == "__main__":
    main()


# python bqa/bqa_rerank.py --weights weights/yolov8m_refined_v4_baseline.pt --input path/to/one_image.jpg --query "syringe"
# python bqa/bqa_rerank.py --weights runs/train/refined-exp4-yolov8m/weights/best.pt --query "syringe"