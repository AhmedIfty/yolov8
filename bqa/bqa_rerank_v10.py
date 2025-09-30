# bqa/bqa_rerank_v10.py
# -------------------------------------------------------------
# YOLO -> (optionally gate by superclass) -> crop -> CLIP subtype
# Strict GREEN/RED competition + per-crop CSV debug + correct overlays
# -------------------------------------------------------------

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

import open_clip
from ultralytics import YOLO

# ----------------------------
# Subtype definitions/prompts
# ----------------------------
SUBTYPES = [
    # sharp-object
    "knife", "scissors", "fork", "pin", "needle", "razor", "nail",
    # medical
    "mask", "glove", "test tube", "syringe",
]

# Group queries by YOLO superclass for gating
GROUPS = {
    "sharp-object": {"knife", "scissors", "fork", "pin", "needle", "razor", "nail"},
    "medical": {"mask", "glove", "test tube", "syringe"},
}

def required_superclass_for_query(q: str):
    q = q.lower().strip()
    for super_name, members in GROUPS.items():
        if q in members:
            return super_name
    return None  # no gating

CANON = {
    # sharp-object
    "knife":"knife", "dagger":"knife", "chef knife":"knife", "kitchen knife":"knife",
    "scissor":"scissors", "scissors":"scissors",
    "fork":"fork",
    "pin":"pin", "thumbtack":"pin",
    "needle":"needle",
    "razor":"razor", "razor blade":"razor", "blade":"razor",
    "nail":"nail", "construction nail":"nail", "common nail":"nail", "iron nail":"nail", "steel nail":"nail",

    # medical
    "mask":"mask", "face mask":"mask", "surgical mask":"mask",
    "glove":"glove", "gloves":"glove", "medical glove":"glove",
    "test tube":"test tube", "blood tube":"test tube", "vacutainer":"test tube", "lab tube":"test tube",
    "syringe":"syringe", "needle syringe":"syringe",
}

POS = {
    # sharp-object
    "knife": [
        "a kitchen knife with a handle and one cutting blade, no finger holes",
        "a straight elongated cutting blade with handle",
    ],
    "scissors": [
        "two blades with a pivot and finger holes",
        "X-shaped scissors with finger rings",
        "shears with two blades and a screw joint",
    ],
    "fork": [
        "a table fork with multiple metal tines",
        "metal utensil with several prongs",
    ],
    "pin": [
        "a thin straight dressmaking pin with a small spherical head",
        "very slender pin, no flat round head disc",
    ],
    "needle": [
        "a very thin sewing needle with a sharp point and an eye near one end",
        "long slender needle, no flat head",
    ],
    "razor": [
        "a double-edged razor blade: thin rectangular metal with cutouts",
        "a safety razor blade sheet, not a rod",
    ],
    "nail": [
        "a construction nail: long straight cylindrical metal shaft with a flat round head disc and a pointed tip",
        "a metal nail with clearly visible flat head and sharp point, no handle, no finger holes, no tines",
        "a single steel nail lying on a surface, round flat head",
    ],

    # medical
    "mask": [
        "a blue surgical face mask with ear loops",
        "a disposable medical mask with pleats and ear loops",
        "protective face mask used in hospitals",
    ],
    "glove": [
        "a blue disposable medical glove",
        "a latex or nitrile glove for medical use",
        "a single rubber glove with finger shapes",
    ],
    "test tube": [
        "a plastic blood collection test tube with colored cap",
        "a laboratory test tube cylindrical with a cap",
        "a vacutainer blood tube with label",
    ],
    "syringe": [
        "a medical syringe with barrel plunger and needle",
        "a plastic syringe with measurement markings and needle",
        "a disposable syringe with a needle tip",
    ],
}
NEG = {st: [x for x in SUBTYPES if x != st] for st in SUBTYPES}

# ----------------------------
# Utilities
# ----------------------------
def _clean_query(q: str) -> str:
    return "".join(c for c in q if c.isalnum() or c in "-_ ").strip().lower()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def to_pil(img_bgr):
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def to_bgr(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_boxes(image_bgr, boxes_xyxy, labels, color=(0, 200, 255), width=3):
    im = to_pil(image_bgr)
    dr = ImageDraw.Draw(im)
    for (x1, y1, x2, y2), lab in zip(boxes_xyxy, labels):
        dr.rectangle([x1, y1, x2, y2], outline=color, width=width)
        # label background
        txt = str(lab)
        tw, th = dr.textlength(txt), 14  # simple height guess
        pad = 3
        dr.rectangle([x1, y1, x1 + tw + 2 * pad, y1 + th + 2 * pad], fill=(0, 0, 0))
        dr.text((x1 + pad, y1 + pad), txt, fill=color)
    return to_bgr(im)

def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1: x2 = min(w, x1 + 1)
    if y2 <= y1: y2 = min(h, y1 + 1)
    return x1, y1, x2, y2

def pad_box(xyxy, pad_ratio, W, H):
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    px = w * pad_ratio
    py = h * pad_ratio
    return clamp_box(x1 - px, y1 - py, x2 + px, y2 + py, W, H)

def norm01(x: np.ndarray):
    a = np.asarray(x, dtype=np.float32)
    if a.size == 0:
        return a
    mn, mx = a.min(), a.max()
    if mx - mn < 1e-9:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)

# ----------------------------
# CLIP helpers
# ----------------------------
class ClipHelper:
    def __init__(self, device, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k",
                 temp=0.05, perc_q=90, neg_beta=0.30):
        self.device = device
        self.model, self.preprocess, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device).eval()
        self.tokenize = open_clip.get_tokenizer(model_name)
        self.temp = temp
        self.perc_q = perc_q
        self.neg_beta = neg_beta
        # encode text prompts
        self.TXT_POS = {k: self._enc_text(POS[k]) for k in POS}
        self.TXT_NEG = {k: self._enc_text(NEG[k]) for k in NEG}

    def _enc_text(self, prompts):
        with torch.no_grad():
            tok = self.tokenize(prompts).to(self.device)
            z = self.model.encode_text(tok)
            z = z / z.norm(dim=-1, keepdim=True)
        return z  # [T,D]

    def encode_image(self, pil_img: Image.Image):
        x = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            z = self.model.encode_image(x)
            z = z / z.norm(dim=-1, keepdim=True)  # [1,D]
        return z

    def percentile_pool(self, sim, q=None):
        if q is None: q = self.perc_q
        k = max(1, int(sim.size(1) * q / 100))
        return sim.topk(k, dim=1).values.mean(dim=1)  # [N]

    def score_subtype(self, name, img_z, temp=None, neg_beta=None, perc_q=None):
        if temp is None: temp = self.temp
        if neg_beta is None: neg_beta = self.neg_beta
        if perc_q is None: perc_q = self.perc_q
        pos = self.TXT_POS[name]          # [T_pos,D]
        sims_pos = (img_z @ pos.T) / temp # [N,T_pos]
        pos_pool = self.percentile_pool(sims_pos, q=perc_q)  # [N]
        neg = self.TXT_NEG[name]
        sims_neg = (img_z @ neg.T) / temp
        neg_max = sims_neg.max(dim=1).values
        return pos_pool - neg_beta * neg_max  # [N]

    def competition(self, img_z, target, thr=0.20, delta=0.08):
        sibs = [s for s in SUBTYPES if s != target]
        s_t = self.score_subtype(target, img_z)  # [1]
        if len(sibs):
            s_stack = torch.stack([self.score_subtype(s, img_z) for s in sibs], dim=1)  # [1,S]
            s_best = s_stack.max(dim=1).values  # [1]
        else:
            s_best = torch.zeros_like(s_t)

        ok = (s_t >= thr) & ((s_t - s_best) >= delta)  # strict, no relaxation
        # Build numpy outputs
        s_t_np = s_t.detach().cpu().numpy().astype(np.float32)
        s_best_np = s_best.detach().cpu().numpy().astype(np.float32)
        gaps_np = (s_t - s_best).detach().cpu().numpy().astype(np.float32)
        lab = "GREEN" if bool(ok[0]) else "RED"
        return bool(ok[0]), float(s_t_np[0]), float(s_best_np[0]), float(gaps_np[0]), lab

# ----------------------------
# Main pipeline
# ----------------------------
def run_one_image(img_path: Path,
                  yolo: YOLO,
                  ch: ClipHelper,
                  query: str,
                  out_dir: Path,
                  dump_crops: int = 0,
                  dump_stage: str = "postgate",
                  pad_ratio: float = 0.15,
                  fuse_thr: float = 0.45,
                  thr: float = 0.20,
                  delta: float = 0.08):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"[WARN] Cannot read image: {img_path}")
        return
    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # YOLO detect
    res = yolo.predict(source=img_bgr, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        vis = img_bgr.copy()
        vis_out = out_dir / f"{img_path.stem}_{_clean_query(query)}.jpg"
        cv2.imwrite(str(vis_out), vis)
        (out_dir / f"{img_path.stem}_{_clean_query(query)}_crops.csv").write_text(
            "idx,x1,y1,x2,y2,yolo_conf,yolo_cls,yolo_name,s_target,s_best,gap,keep_ok,fused\n"
        )
        return

    xyxy = res.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    conf = res.boxes.conf.detach().cpu().numpy().astype(np.float32)
    cls_ids = res.boxes.cls.detach().cpu().numpy().astype(int)
    # names from model
    id2name = yolo.model.names if hasattr(yolo, "model") else res.names
    yolo_names = [id2name[int(i)] for i in cls_ids]

    # ---- Superclass gating ----
    target_query = CANON.get(query.lower(), query.lower())
    needed_super = required_superclass_for_query(target_query)  # 'medical', 'sharp-object', or None
    mask_gating = np.ones(len(xyxy), dtype=bool)
    if needed_super is not None:
        mask_gating = np.array([ (nm == needed_super) for nm in yolo_names ], dtype=bool)

    # Per-crop evaluation
    kept_boxes_draw, kept_labels, kept_scores = [], [], []   # draw with YOLO box
    rows_debug = []
    crops_dir = out_dir / f"{img_path.stem}_{_clean_query(query)}_crops"
    if dump_crops:
        ensure_dir(crops_dir)

    for idx, (b, c, name, use_it) in enumerate(zip(xyxy, conf, yolo_names, mask_gating)):
        # If gating says "no", skip CLIP and mark RED in CSV for transparency
        if not use_it:
            rows_debug.append({
                "idx": idx,
                "x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3]),
                "yolo_conf": float(c),
                "yolo_cls": int(cls_ids[idx]),
                "yolo_name": name,
                "s_target": float("nan"),
                "s_best": float("nan"),
                "gap": float("nan"),
                "keep_ok": False,
                "fused": float((1.0 - fuse_thr) * c),
            })
            continue

        # --- boxes ---
        # draw box = original YOLO box (compact)
        x1d, y1d, x2d, y2d = clamp_box(b[0], b[1], b[2], b[3], W, H)

        # crop box = padded for CLIP
        x1c, y1c, x2c, y2c = pad_box(b, pad_ratio, W, H)

        # crop for CLIP
        crop = img_rgb[int(y1c):int(y2c), int(x1c):int(x2c), :]
        crop_pil = Image.fromarray(crop)

        # encode + compete
        z = ch.encode_image(crop_pil)
        ok, s_t, s_best, gap, _ = ch.competition(z, target_query, thr=thr, delta=delta)

        # fused score (kept for CSV completeness)
        fused = float(fuse_thr * norm01(np.array([s_t]))[0] + (1.0 - fuse_thr) * c)

        # dump crop(s)
        if dump_crops and (dump_stage == "pregate" or (dump_stage == "postgate" and ok)):
            crop_path = crops_dir / f"{idx:03d}_{'KEEP' if ok else 'DROP'}_T{s_t:.3f}_O{s_best:.3f}_d{gap:.3f}.jpg"
            cv2.imwrite(str(crop_path), cv2.cvtColor(np.array(crop_pil), cv2.COLOR_RGB2BGR))

        # debug row (use crop box for recorded coords to match the crop; drawing uses YOLO box)
        rows_debug.append({
            "idx": idx,
            "x1": float(x1c), "y1": float(y1c), "x2": float(x2c), "y2": float(y2c),
            "yolo_conf": float(c),
            "yolo_cls": int(cls_ids[idx]),
            "yolo_name": name,
            "s_target": float(s_t),
            "s_best": float(s_best),
            "gap": float(gap),
            "keep_ok": bool(ok),
            "fused": float(fused),
        })

        # keep only GREEN
        if ok:
            kept_boxes_draw.append([x1d, y1d, x2d, y2d])  # draw with YOLO compact box
            kept_labels.append(target_query)  # only the query text
            kept_scores.append(fused)

    # Save CSV
    import pandas as pd
    df = pd.DataFrame(
        rows_debug,
        columns=["idx","x1","y1","x2","y2","yolo_conf","yolo_cls","yolo_name","s_target","s_best","gap","keep_ok","fused"]
    )
    df.to_csv(out_dir / f"{img_path.stem}_{_clean_query(query)}_crops.csv", index=False)

    # Draw & save
    vis = img_bgr.copy()
    if len(kept_boxes_draw):
        vis = draw_boxes(vis, kept_boxes_draw, kept_labels)
    vis_out = out_dir / f"{img_path.stem}_{_clean_query(query)}.jpg"
    cv2.imwrite(str(vis_out), vis)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--input", type=str, required=True, help="image file or folder")
    ap.add_argument("--query", type=str, required=True)

    # YOLO
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)

    # CLIP knobs (match your Colab defaults)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--perc_q", type=int, default=90)
    ap.add_argument("--neg_beta", type=float, default=0.30)
    ap.add_argument("--thr", type=float, default=0.20)
    ap.add_argument("--delta", type=float, default=0.08)

    # Fusion & crops
    ap.add_argument("--fuse_thr", type=float, default=0.45, help="lambda for CLIP in fused score")
    ap.add_argument("--pad", type=float, default=0.15, help="pad ratio around YOLO box before crop")
    ap.add_argument("--dump_crops", type=int, default=1)
    ap.add_argument("--dump_stage", type=str, default="postgate", choices=["pregate","postgate"])
    ap.add_argument("--out", type=str, default="bqa/bqa_result")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CLIP helper (fully respects CLI knobs)
    ch = ClipHelper(
        device=device,
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        temp=args.temp,
        perc_q=args.perc_q,
        neg_beta=args.neg_beta,
    )

    # YOLO
    yolo = YOLO(args.weights)
    yolo.overrides["imgsz"] = args.imgsz
    yolo.overrides["conf"] = args.conf
    yolo.overrides["iou"] = args.iou

    in_path = Path(args.input)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    paths = []
    if in_path.is_dir():
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
            paths += list(in_path.glob(ext))
        paths = sorted(paths)
    else:
        paths = [in_path]

    if not paths:
        print(f"[ERR] No images found in {in_path}")
        return

    q_raw = args.query.strip().lower()
    q = CANON.get(q_raw, q_raw)

    for p in paths:
        run_one_image(
            img_path=p,
            yolo=yolo,
            ch=ch,
            query=q,
            out_dir=out_dir,
            dump_crops=args.dump_crops,
            dump_stage=args.dump_stage,
            pad_ratio=args.pad,
            fuse_thr=args.fuse_thr,
            thr=args.thr,
            delta=args.delta,
        )

    print(f"[OK] Done. Results saved under: {out_dir.resolve()}")

if __name__ == "__main__":
    main()


# python bqa/bqa_rerank_v10.py --weights weights/yolov8m_refined_v4_baseline.pt --input bqa/bqa_test --query "syringe" --fuse_thr 0.45 --temp 0.05 --perc_q 90 --thr 0.20 --delta 0.08 --neg_beta 0.30 --dump_crops 1 --dump_stage postgate
