import argparse
from pathlib import Path
import re
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw
import torch
from ultralytics import YOLO
import open_clip
from tqdm import tqdm

# -----------------------------
# Defaults (updated)
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMGZ = 640
YOLO_CONF = 0.15
YOLO_IOU = 0.50
TOPK = 75
LAMBDA = 0.55  # fuse_lambda default
FUSE_THR = 0.55  # fuse_thr default
# --- CLIP scoring tunables ---
TEMP = 0.05
PERC_Q = 90
MARGIN = 0.10
NEG_BETA = 0.30

# === Dataset classes (order from your data.yaml) ===
CLASS_NAMES = ['glass', 'medical', 'metal', 'organic', 'paper', 'plastic', 'sharp-object']
NAME2ID = {n: i for i, n in enumerate(CLASS_NAMES)}

# --- Prototype scoring ---
PROTO_W = 0.35  # weight for prototype similarity in the clip score (tune 0.2–0.5)
PROTO_DELTA = 0.05  # target proto must beat other protos by this margin


def load_prototypes(root="bqa/prototypes"):
    bank = {}
    root = Path(root)
    for group in ["sharp-object", "medical"]:
        gdir = root / group
        if not gdir.exists():
            continue
        bank[group] = {}
        for npy in gdir.glob("*.npy"):
            bank[group][npy.stem] = np.load(npy).astype(np.float32)  # [K,D], already L2-normalized
    return bank


def proto_score(img_emb: torch.Tensor, proto_mat: np.ndarray, topk=3):
    # img_emb: [N,D] torch, L2-normalized; proto_mat: [K,D] numpy (L2-normalized)
    P = torch.from_numpy(proto_mat).to(img_emb.device)
    sims = img_emb @ P.T  # [N,K]
    k = min(topk, sims.size(1))
    return sims.topk(k, dim=1).values.mean(dim=1)  # [N]


# ---------------------------------------------------
# Query routing config (edit/extend these in one place)
# ---------------------------------------------------
QUERY_ROUTING = {
    "sharp_hazard": {
        "keywords": ["knife", "scissor", "fork", "pin", "needle", "razor", "sharp"],
        "allow": ["sharp-object"],
        "pos_prompts": [
            "a photo of a {}",
            "a close up of a {}",
            "an image of a {}",
            "a {} on a surface",
        ],
        "neg_prompts": {
            "knife": ["fork", "spoon", "metal rod", "test tube", "syringe"],
            "scissors": ["tongs", "tweezers", "calipers", "stapler"],
            "fork": ["spoon", "knife", "comb"],
            "pin": ["nail", "screw", "wire", "staple"],
            "needle": ["pin", "wire", "staple", "syringe"],
            "razor": ["box cutter", "utility knife", "scraper"],
        }
    },
    "medical_hazard": {
        "keywords": ["syringe", "test tube", "glove", "mask", "vial", "bandage"],
        "allow": ["medical"],
        "pos_prompts": [
            "a photo of a {}",
            "a close up of a {}",
            "an image of a {}",
            "a {} on a surface",
            "medical equipment: a {}",
        ],
        "neg_prompts": {
            "syringe": ["test tube", "pipette", "pen", "thermometer", "vial"],
            "test tube": ["syringe", "vial", "pipette", "bottle"],
            "gloves": ["plastic bag", "cloth", "mask", "bandage"],
            "mask": ["glove", "cloth", "paper", "tissue", "bandage"],
        }
    },
    # Can add "glass_bottle", "plastic_bottle" etc. similarly
}


def _clean_query(q: str):
    return re.sub(r'[^a-z0-9\s-]', '', q.lower().strip())


def build_query_tokens(tokenizer, query: str, route: dict):
    q = _clean_query(query)
    pos_templates = route.get("pos_prompts", ["a photo of {}"])
    pos_list = [t.format(q) for t in pos_templates]

    neg_list = []
    if "neg_prompts" in route:
        for subtype, negs in route["neg_prompts"].items():
            if subtype.replace("_", " ") in q:
                neg_list = negs
                break

    pos_tokens = tokenizer(pos_list)
    neg_tokens = tokenizer(neg_list) if len(neg_list) > 0 else None
    return pos_tokens, neg_tokens


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
def clip_scores_for_boxes_v2(clip_model, preprocess, pos_tokens, neg_tokens,
                             image_bgr, boxes_xyxy,
                             temp=0.05, perc_q=90, beta=0.35, margin=0.10):
    txt_pos = clip_model.encode_text(pos_tokens.to(DEVICE))
    txt_pos = txt_pos / txt_pos.norm(dim=-1, keepdim=True)

    if neg_tokens is not None:
        txt_neg = clip_model.encode_text(neg_tokens.to(DEVICE))
        txt_neg = txt_neg / txt_neg.norm(dim=-1, keepdim=True)
    else:
        txt_neg = None

    H, W = image_bgr.shape[:2]
    crops, valid_idx = [], []
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        if x2 <= x1 or y2 <= y1: continue
        crop = image_bgr[max(0, y1):min(H, y2), max(0, x1):min(W, x2), ::-1]
        crops.append(Image.fromarray(crop))
        valid_idx.append(i)

    final_scores = np.zeros(len(boxes_xyxy), dtype=np.float32)
    pos_scores = np.zeros(len(boxes_xyxy), dtype=np.float32)
    neg_scores = np.zeros(len(boxes_xyxy), dtype=np.float32)

    if not crops:
        return final_scores, pos_scores, neg_scores, torch.empty(0)

    batch = torch.cat([preprocess(c).unsqueeze(0) for c in crops]).to(DEVICE)
    img = clip_model.encode_image(batch)
    img = img / img.norm(dim=-1, keepdim=True)

    sims_pos = (img @ txt_pos.T).float()
    sims_pos_scaled = sims_pos / temp

    pos_q = torch.quantile(sims_pos_scaled, perc_q / 100.0, dim=1)

    if txt_neg is not None:
        sims_neg = (img @ txt_neg.T).float()
        sims_neg_scaled = sims_neg / temp
        neg_max, _ = sims_neg_scaled.max(dim=1)

        final = pos_q - beta * neg_max
        # Margin filter: score drops if pos is not clearly better than neg
        final = torch.where(pos_q > neg_max + margin / temp, final, final - 2.0 / temp)
    else:
        final = pos_q
        neg_max = torch.zeros_like(final)

    for j, idx in enumerate(valid_idx):
        final_scores[idx] = float(final[j].item())
        pos_scores[idx] = float(pos_q[j].item())
        neg_scores[idx] = float(neg_max[j].item())

    return final_scores, pos_scores, neg_scores, img.cpu()


def norm01(x: np.ndarray):
    if x.size == 0: return x
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-9) if mx > mn else np.zeros_like(x)


def draw_boxes(im, boxes, scores, label, color):
    dr = ImageDraw.Draw(im)
    for (x1, y1, x2, y2), sc in zip(boxes, scores):
        dr.rectangle([x1, y1, x2, y2], outline=color, width=3)
        dr.text((x1 + 3, y1 + 3), f"{label} {sc:.2f}", fill=color)


def run_on_image(det, clip_model, preprocess, tokenizer,
                 img_path: Path, query: str, proto_bank: dict,
                 out_dir: Path = None, topk: int = TOPK, fuse_lambda: float = LAMBDA, fuse_thr: float = FUSE_THR,
                 yolo_conf: float = YOLO_CONF, yolo_iou: float = YOLO_IOU, imgsz: int = IMGZ,
                 neg_beta: float = NEG_BETA):
    im_bgr = cv2.imread(str(img_path))
    if im_bgr is None: raise FileNotFoundError(img_path)

    q_lower = _clean_query(query)
    route = None
    for group_name, cfg in QUERY_ROUTING.items():
        if any(kw in q_lower for kw in cfg['keywords']):
            route = cfg
            route["group_name"] = group_name
            break
    if route is None:
        print(f"Warning: No route found for query '{query}', using defaults.")
        route = {"allow": [], "group_name": "unknown"}

    r = det.predict(source=str(img_path), imgsz=imgsz, conf=yolo_conf, iou=yolo_iou, verbose=False)[0]
    if r.boxes is None or r.boxes.xyxy.numel() == 0:
        return {"image": str(img_path), "n_proposals": 0, "n_kept": 0}

    boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
    yolo_scores = r.boxes.conf.cpu().numpy().astype(np.float32)
    yolo_cls = r.boxes.cls.cpu().numpy().astype(int)

    # STRICT gate: if nothing from allowed classes, stop early
    if route and "allow" in route and len(route["allow"]) > 0:
        allow_ids = np.array([NAME2ID[n] for n in route["allow"] if n in NAME2ID], dtype=int)
        gate = np.isin(yolo_cls, allow_ids)
        if not gate.any():
            return {"image": str(img_path), "n_proposals": len(yolo_cls), "n_kept": 0}
        boxes, yolo_scores, yolo_cls = boxes[gate], yolo_scores[gate], yolo_cls[gate]

    idx = np.argsort(-yolo_scores)[:topk]
    boxes, yolo_scores = boxes[idx], yolo_scores[idx]

    pos_tokens, neg_tokens = build_query_tokens(tokenizer, query, route)

    s_clip, _, _, img_emb_t = clip_scores_for_boxes_v2(
        clip_model, preprocess, pos_tokens, neg_tokens, im_bgr, boxes,
        temp=TEMP, perc_q=PERC_Q, beta=neg_beta, margin=MARGIN
    )

    # --- Prototype enhancement (if we have a bank for this group/subtype) ---
    proto_margin_mask = None
    group_name = route["group_name"]
    if group_name in ("sharp_hazard", "medical_hazard") and img_emb_t.numel() > 0:
        q = _clean_query(query)
        subtype_key = None
        SUBTOK = {
            "sharp_hazard": ["knife", "scissors", "fork", "pin", "needle", "razor"],
            "medical_hazard": ["syringe", "test_tube", "gloves", "mask", "vial", "bandage"]
        }
        for t in SUBTOK[group_name]:
            if t.replace("_", " ") in q:
                subtype_key = t
                break

        group_dir = "sharp-object" if group_name == "sharp_hazard" else "medical"
        if group_dir in proto_bank:
            # compute sims to all available subtypes
            sims_map = {}
            for st, bank in proto_bank[group_dir].items():
                if bank.shape[0] == 0:
                    continue
                sims_map[st] = proto_score(img_emb_t, bank, topk=3)  # torch [N]

            if subtype_key in sims_map and len(sims_map) > 1:
                target = sims_map[subtype_key]
                others = torch.stack([v for k, v in sims_map.items() if k != subtype_key], dim=1)
                other_max = others.max(dim=1).values
                proto_margin_mask = (target - other_max) >= PROTO_DELTA

                # add prototype bonus only; margin mask will be applied in final keep
                s_clip = s_clip + PROTO_W * norm01(target.cpu().numpy())
            else:
                proto_margin_mask = torch.ones(img_emb_t.size(0), dtype=torch.bool)

            # keep this mask to combine with fused threshold later
            proto_margin_mask = proto_margin_mask.cpu().numpy()

    s_clip_n = norm01(s_clip)
    s_yolo_n = norm01(yolo_scores)
    fused = fuse_lambda * s_clip_n + (1.0 - fuse_lambda) * s_yolo_n

    if q_lower in {"syringe", "needle", "knife", "pin"}:
        MIN_AR = 2.4
        w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        ar = np.maximum(w, h) / np.maximum(1.0, np.minimum(w, h))
        fused = np.where(ar >= MIN_AR, fused, 0.85 * fused)

    keep = fused >= fuse_thr
    keep = keep & ((proto_margin_mask if proto_margin_mask is not None else np.ones_like(keep)).astype(bool))
    boxes_out, scores_out = boxes[keep], fused[keep]
    order = np.argsort(-scores_out)
    boxes_out, scores_out = boxes_out[order], scores_out[order]

    im = Image.fromarray(cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB))
    if len(boxes_out) > 0:
        draw_boxes(im, boxes_out, scores_out, label=query, color=(0, 200, 255))

    rec = {"image": str(img_path), "n_proposals": len(boxes), "n_kept": len(boxes_out),
           "boxes": boxes_out.tolist(), "scores": scores_out.tolist()}

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_img = out_dir / f"{img_path.stem}_{_clean_query(query)}.jpg"
        im.save(out_img)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to YOLO best.pt")
    ap.add_argument("--input", required=True, help="image file or folder")
    ap.add_argument("--query", required=True, help="e.g., 'syringe', 'knife'")
    ap.add_argument("--output", default="bqa/bqa_result_v5", help="folder for visualizations")
    ap.add_argument("--clip", default="ViT-B-32", help="open-clip model name")
    ap.add_argument("--clip_pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--fuse_lambda", type=float, default=LAMBDA)
    ap.add_argument("--fuse_thr", type=float, default=FUSE_THR)
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--yolo_conf", type=float, default=YOLO_CONF)
    ap.add_argument("--yolo_iou", type=float, default=YOLO_IOU)
    ap.add_argument("--imgsz", type=int, default=IMGZ)
    ap.add_argument("--neg_beta", type=float, default=NEG_BETA, help="Weight for negative prompts")
    args = ap.parse_args()

    det, clip_model, preprocess, tokenizer = load_models(
        args.weights, args.clip, args.clip_pretrained
    )
    proto_bank = load_prototypes("bqa/prototypes")

    in_path = Path(args.input)
    out_dir = Path(args.output)

    if in_path.is_file():
        rec = run_on_image(det, clip_model, preprocess, tokenizer, in_path, args.query, proto_bank,
                           out_dir, args.topk, args.fuse_lambda, args.fuse_thr,
                           args.yolo_conf, args.yolo_iou, args.imgsz, args.neg_beta)
        print(rec)
    else:
        imgs = sorted(list(in_path.glob("*.jpg")) + list(in_path.glob("*.png")) + list(in_path.glob("*.jpeg")))
        print(f"[INFO] Found {len(imgs)} images in {in_path}")
        out_csv = out_dir / f"bqa_{_clean_query(args.query).replace(' ', '_')}.csv"
        rows = []
        for p in tqdm(imgs, desc="BQA"):
            rec = run_on_image(det, clip_model, preprocess, tokenizer, p, args.query, proto_bank,
                               out_dir, args.topk, args.fuse_lambda, args.fuse_thr,
                               args.yolo_conf, args.yolo_iou, args.imgsz, args.neg_beta)
            rows.append(rec)
        try:
            import pandas as pd
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"[OK] Wrote {out_csv}")
        except Exception as e:
            print(f"[WARN] could not write csv: {e}")


if __name__ == "__main__":
    main()

