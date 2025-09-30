import argparse, re
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image, ImageDraw
from ultralytics import YOLO
import open_clip
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMGZ = 640
YOLO_CONF = 0.15
YOLO_IOU  = 0.50
TOPK      = 150               # show me everything while debugging
LAMBDA    = 0.5
FUSE_THR  = 0.40              # a bit friendlier
TEMP      = 0.05
PERC_Q    = 90
NEG_BETA  = 0.30
MARGIN    = 0.10
THR       = 0.16              # softened
DELTA     = 0.05              # softened

CLASS_NAMES = ['glass','medical','metal','organic','paper','plastic','sharp-object']
NAME2ID = {n:i for i,n in enumerate(CLASS_NAMES)}


# ---------------------------------------------------
# Query routing config (+ subtype libraries)
# (same content as v4, trimmed comments)
# ---------------------------------------------------
QUERY_GROUPS = {
    "medical_hazard": {
        "keywords": [
            "syringe", "test tube", "vial", "lab tube", "mask", "face mask",
            "surgical mask", "glove", "gloves", "cotton", "bandage",
            "iv bag", "saline bag", "medicine packet", "drug packet", "blood tube"
        ],
        "allow": ["medical"],
        "strict": True,
        "pos": [],
        "neg": ["knife", "needle", "scissor", "axe", "blade", "screw", "metal rod"],
        "subtypes": {
            "syringe": {
                "keywords": ["syringe", "injection syringe", "hypodermic syringe"],
                "pos": [
                    "plastic barrel with plunger",
                    "needle attached to barrel",
                    "graduated markings on barrel",
                    "narrow needle and protective cap",
                    "plunger rod with thumb rest",
                    "transparent cylindrical barrel",
                ],
                "neg": ["test tube", "pipette", "dropper", "pen", "marker", "vial", "lab tube"]
            },
            "test_tube": {
                "keywords": ["test tube", "lab tube", "blood tube"],
                "pos": [
                    "cylindrical glass tube",
                    "open top rounded bottom",
                    "no handle no plunger",
                    "placed in test tube rack",
                    "thin glass laboratory tube",
                ],
                "neg": ["syringe", "pipette", "dropper", "vial", "bottle", "glass rod", "pen"]
            },
            "vial": {
                "keywords": ["vial"],
                "pos": [
                    "small glass bottle",
                    "short neck with rubber stopper",
                    "sealed cap or crimp top",
                    "cylindrical container for medicine",
                ],
                "neg": ["test tube", "syringe", "bottle", "pipette", "dropper"]
            },
            "gloves": {
                "keywords": ["glove", "gloves"],
                "pos": [
                    "pair of disposable gloves",
                    "five finger shape",
                    "latex or nitrile texture",
                    "rolled cuff at wrist",
                ],
                "neg": ["mask", "bandage", "cotton", "tissue", "paper"]
            },
            "mask": {
                "keywords": ["mask", "face mask", "surgical mask"],
                "pos": [
                    "rectangular pleated face mask",
                    "ear loops on both sides",
                    "three ply fabric layers",
                    "nose bridge strip",
                ],
                "neg": ["gloves", "bandage", "cotton", "tissue"]
            },
            "bandage": {
                "keywords": ["bandage", "cotton"],
                "pos": [
                    "white cotton roll or gauze strip",
                    "soft fibrous texture",
                    "medical dressing material",
                ],
                "neg": ["mask", "gloves", "tissue", "paper towel"]
            },
        },
    },

    "sharp_hazard": {
        "keywords": [
            "scissor", "scissors", "knife", "needle", "pin", "axe", "ax",
            "fork", "blade", "screw", "box cutter", "cutter", "razor"
        ],
        "allow": ["sharp-object"],
        "strict": True,
        "pos": [],
        "neg": ["syringe", "test tube", "mask", "glove"],
        "subtypes": {
            "knife": {
                "keywords": ["knife", "chef knife", "kitchen knife", "butter knife"],
                "pos": [
                    "single long blade with a handle",
                    "handle plus one cutting blade, no finger holes",
                    "elongated knife silhouette with blade and handle",
                ],
                "neg": [
                    # ultra-close distractors
                    "razor blade", "double edged razor blade", "box cutter", "paper cutter",
                    "scissors", "scissor",
                    "pin", "safety pin", "needles", "needle", "nail", "thumbtack", "push pin",
                    "fork", "axe"
                ],
            },
            "scissors": {
                "keywords": ["scissor", "scissors"],
                "pos": [
                    "two blades connected by a pivot screw",
                    "x-shaped scissors with finger holes in handles",
                    "pair of blades joined at a pivot with looped handles"
                ],
                "neg": [
                    "knife", "razor blade", "box cutter", "paper cutter",
                    "fork", "pin", "safety pin", "needle", "nail", "screw", "wire cutters"
                ],
            },
            "fork": {
                "keywords": ["fork", "table fork", "dining fork"],
                "pos": [
                    "metal utensil with multiple prongs or tines",
                    "table fork with four tines",
                    "fork head with parallel tines and a handle"
                ],
                "neg": [
                    "knife", "scissors", "razor blade",
                    "pin", "needle", "nail", "safety pin", "spoon", "chopsticks"
                ],
            },
            "pin": {
                "keywords": ["pin", "push pin", "thumbtack", "tack"],
                "pos": [
                    "thin needle-like shaft with a small round head",
                    "very slender straight metal pin, no handle"
                ],
                "neg": [
                    "needle with an eye", "sewing needle",
                    "nail with a flat head", "screw", "bolt",
                    "knife", "razor blade", "scissors", "fork", "safety pin"
                ],
            },
            "needle": {
                "keywords": ["needle", "sewing needle"],
                "pos": [
                    "very slender sewing needle with an eye",
                    "long straight needle with a tiny eye at one end"
                ],
                "neg": [
                    "pin with a round head", "thumbtack", "push pin",
                    "nail with flat head", "knife", "razor blade", "scissors", "fork", "safety pin"
                ],
            },
            "razor": {
                "keywords": ["razor", "razor blade", "safety razor blade"],
                "pos": [
                    "thin rectangular razor blade with central cutouts",
                    "double edged safety razor blade",
                    "rectangular metal blade with notches"
                ],
                "neg": [
                    "knife", "box cutter", "paper cutter",
                    "scissors", "fork",
                    "pin", "safety pin", "needle", "nail"
                ],
            },
            "nail": {
                "keywords": ["nail"],
                "pos": [
                    "metal nail with a flat head and pointed tip",
                    "construction nail with cylindrical shaft"
                ],
                "neg": [
                    "pin with small ball head", "sewing needle with eye",
                    "screw with threads", "bolt",
                    "knife", "razor blade", "scissors", "fork", "safety pin"
                ],
            },
            "safety_pin": {
                "keywords": ["safety pin"],
                "pos": [
                    "curved safety pin with closing clasp",
                    "pin with a looped end and spring clasp"
                ],
                "neg": [
                    "straight pin", "thumbtack", "needle", "nail",
                    "knife", "razor blade", "scissors", "fork"
                ],
            },

        },
    },

    # demo bottle groups unchanged from v4 …
    "plastic_bottle": {
        "keywords": ["plastic bottle", "water bottle", "drink bottle", "pet bottle", "mineral water bottle"],
        "allow": ["plastic"],
        "strict": False,
        "pos": [
            "clear ribbed plastic bottle",
            "PET bottle with screw cap",
            "lightweight transparent polymer bottle",
            "thin flexible plastic walls",
        ],
        "neg": ["glass bottle", "wine bottle", "beer bottle", "glass"]
    },
    "glass_bottle": {
        "keywords": ["glass bottle", "beer bottle", "wine bottle"],
        "allow": ["glass"],
        "strict": False,
        "pos": [
            "thick glass bottle",
            "long neck with lip",
            "rigid reflective glass surface",
            "heavy glass base often punted",
        ],
        "neg": ["plastic bottle", "mineral water bottle", "pet bottle", "plastic"]
    },
}

# ----------------- helpers -----------------
_RX_WS = re.compile(r"\s+")
def _clean_query(q:str)->str:
    q = q.lower().strip()
    for t in ["a picture of","picture of","photo of","an image of","image of","a photo of"]:
        q = q.replace(t,"")
    return _RX_WS.sub(" ", q).strip()

def match_group_and_subtype(query: str):
    q = _clean_query(query)
    for gname, cfg in QUERY_GROUPS.items():
        if any(kw in q for kw in cfg["keywords"]):
            allow_ids = {NAME2ID[n] for n in cfg["allow"] if n in NAME2ID}
            route = dict(group_name=gname, allow_ids=allow_ids, strict=bool(cfg.get("strict",False)),
                         pos_prompts=cfg.get("pos",[]) or [q], neg_prompts=cfg.get("neg",[]) or [], cfg=cfg)
            subtype_target, siblings = None, []
            stlib = cfg.get("subtypes",{}) or {}
            for st_name, st_cfg in stlib.items():
                if any(skw in q for skw in st_cfg.get("keywords",[st_name])):
                    subtype_target = st_name; break
            if subtype_target:
                siblings = [s for s in stlib.keys() if s!=subtype_target]
            return route, subtype_target, siblings
    return dict(group_name=None, allow_ids=None, strict=False,
                pos_prompts=[_clean_query(query)], neg_prompts=[], cfg=None), None, []

def load_models(yolo_weights:str, clip_name="ViT-B-32", clip_pretrained="laion2b_s34b_b79k"):
    det = YOLO(yolo_weights)
    model, _, preprocess = open_clip.create_model_and_transforms(clip_name, pretrained=clip_pretrained, device=DEVICE)
    tokenizer = open_clip.get_tokenizer(clip_name)
    model.eval()
    return det, model, preprocess, tokenizer

def _expand_rect(x1,y1,x2,y2,W,H,pad_xy=(0.06,0.06)):
    bw, bh = x2-x1, y2-y1
    px, py = pad_xy[0]*bw, pad_xy[1]*bh
    ex1 = int(max(0, np.floor(x1 - px))); ey1 = int(max(0, np.floor(y1 - py)))
    ex2 = int(min(W, np.ceil(x2 + px)));  ey2 = int(min(H, np.ceil(y2 + py)))
    if ex2<=ex1: ex2 = min(W, ex1+4)
    if ey2<=ey1: ey2 = min(H, ey1+4)
    return ex1,ey1,ex2,ey2

def _letterbox_to_square(img_bgr, min_side=64, pad_color=(114,114,114)):
    h,w = img_bgr.shape[:2]
    side = max(h,w,min_side)
    top  = (side-h)//2; bottom = side-h-top
    left = (side-w)//2; right  = side-w-left
    return cv2.copyMakeBorder(img_bgr, top,bottom,left,right,
                              borderType=cv2.BORDER_CONSTANT, value=pad_color)

@torch.no_grad()
def encode_crops(clip_model, preprocess, image_bgr, boxes_xyxy,
                 pad_xy=(0.06,0.06), do_square=True,
                 dump_crops=False, dump_dir:Path=None, image_stem:str="img"):
    H,W = image_bgr.shape[:2]
    crops, valid_idx = [], []
    for i,(x1,y1,x2,y2) in enumerate(boxes_xyxy):
        x1=float(x1); y1=float(y1); x2=float(x2); y2=float(y2)
        if x2<=x1 or y2<=y1: continue
        ex1,ey1,ex2,ey2 = _expand_rect(x1,y1,x2,y2,W,H,pad_xy=pad_xy)
        crop = image_bgr[ey1:ey2, ex1:ex2]
        if crop.size==0: continue
        if do_square: crop = _letterbox_to_square(crop, min_side=64)
        if dump_crops and dump_dir is not None:
            dd = dump_dir / image_stem
            dd.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dd / f"{i:03d}.jpg"), crop)
        crops.append(Image.fromarray(crop[:,:,::-1]))   # BGR->RGB
        valid_idx.append(i)
    if not crops: return None,[]
    batch = torch.cat([preprocess(c).unsqueeze(0) for c in crops], dim=0).to(DEVICE)
    zimg = clip_model.encode_image(batch)
    zimg = zimg / zimg.norm(dim=-1, keepdim=True)
    return zimg, valid_idx

@torch.no_grad()
def encode_text(clip_model, tokenizer, prompts):
    if not prompts: return None
    tok = tokenizer(prompts).to(DEVICE)
    z = clip_model.encode_text(tok)
    return z / z.norm(dim=-1, keepdim=True)

def percentile_pool(x, q=PERC_Q, dim=1):
    k = max(1, int(x.size(dim)*q/100))
    return x.topk(k, dim=dim).values.mean(dim=dim)

def draw_boxes(image_bgr, boxes, scores, label="query", color=(0,200,255)):
    im = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(im)
    for (x1,y1,x2,y2), sc in zip(boxes, scores):
        dr.rectangle([x1,y1,x2,y2], outline=color, width=3)
        dr.text((x1+3,y1+3), f"{label} {sc:.2f}", fill=color)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

def norm01(x:np.ndarray):
    if x.size==0: return x
    mn,mx = x.min(), x.max()
    return np.zeros_like(x) if mx<=mn+1e-9 else (x-mn)/(mx-mn+1e-9)

@torch.no_grad()
def subtype_competition(clip_model, preprocess, tokenizer, image_bgr, boxes_xyxy,
                        stlib, target_name, siblings,
                        dump_crops, dump_dir, image_stem):
    zimg, valid_idx = encode_crops(clip_model, preprocess, image_bgr, boxes_xyxy,
                                   dump_crops=dump_crops, dump_dir=dump_dir, image_stem=image_stem)
    N = len(boxes_xyxy)
    s_target = np.zeros((N,), np.float32); s_best = np.zeros_like(s_target); ok = np.zeros((N,), bool)
    if zimg is None: return ok, s_target, s_best

    def text_feats(name):
        cfg = stlib.get(name, {})
        zp = encode_text(clip_model, tokenizer, cfg.get("pos",[]) or [name])
        zn = encode_text(clip_model, tokenizer, cfg.get("neg",[])) if (cfg.get("neg") and NEG_BETA>0) else None
        return zp, zn

    zt_pos, zt_neg = text_feats(target_name)
    sims_pos_t = (zimg @ zt_pos.T) / TEMP
    pos_t = percentile_pool(sims_pos_t, q=PERC_Q, dim=1)
    neg_t = (zimg @ zt_neg.T).max(dim=1).values if zt_neg is not None else torch.zeros_like(pos_t)
    s_t = pos_t - NEG_BETA*neg_t

    if siblings:
        sib_list = []
        for sname in siblings:
            zp, zn = text_feats(sname)
            sp = percentile_pool((zimg @ zp.T)/TEMP, q=PERC_Q, dim=1)
            sn = (zimg @ zn.T).max(dim=1).values if zn is not None else torch.zeros_like(sp)
            sib_list.append(sp - NEG_BETA*sn)
        s_others = torch.stack(sib_list, dim=1).max(dim=1).values if sib_list else torch.zeros_like(s_t)
    else:
        s_others = torch.zeros_like(s_t)

    mask = (s_t >= THR) & ((s_t - s_others) >= DELTA)
    for j, idx in enumerate(valid_idx):
        s_target[idx] = float(s_t[j].item())
        s_best[idx]   = float(s_others[j].item())
        ok[idx]       = bool(mask[j].item())
    return ok, s_target, s_best

@torch.no_grad()
def clip_simple(clip_model, preprocess, tokenizer, image_bgr, boxes_xyxy,
                pos_prompts, neg_prompts, dump_crops, dump_dir, image_stem):
    zpos = encode_text(clip_model, tokenizer, pos_prompts)
    zneg = encode_text(clip_model, tokenizer, neg_prompts) if (neg_prompts and NEG_BETA>0) else None
    zimg, valid_idx = encode_crops(clip_model, preprocess, image_bgr, boxes_xyxy,
                                   dump_crops=dump_crops, dump_dir=dump_dir, image_stem=image_stem)
    out_clip = np.zeros((len(boxes_xyxy),), np.float32)
    out_pos  = np.zeros_like(out_clip); out_neg = np.zeros_like(out_clip)
    if zimg is None or zpos is None: return out_clip, out_pos, out_neg
    sp = percentile_pool((zimg @ zpos.T)/TEMP, q=PERC_Q, dim=1)
    sn = (zimg @ zneg.T).max(dim=1).values if zneg is not None else torch.zeros_like(sp)
    s  = sp - NEG_BETA*sn
    for j, idx in enumerate(valid_idx):
        out_clip[idx] = float(s[j].item()); out_pos[idx]=float(sp[j].item()); out_neg[idx]=float(sn[j].item())
    return out_clip, out_pos, out_neg

def _save_rec(img_path, query, out_dir, im_bgr, boxes_out, scores_out,
              counts, pre_overlay=None):
    if len(boxes_out)>0:
        vis = draw_boxes(im_bgr, boxes_out, scores_out, label=query)
    else:
        vis = im_bgr
    rec = {
        "image": str(img_path),
        **counts,
        "boxes": (boxes_out.tolist() if hasattr(boxes_out,"tolist") else []),
        "scores": (scores_out.tolist() if hasattr(scores_out,"tolist") else [])
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        if pre_overlay is not None:
            cv2.imwrite(str(out_dir / f"{img_path.stem}_{_clean_query(query)}_pre.jpg"), pre_overlay)
        cv2.imwrite(str(out_dir / f"{img_path.stem}_{_clean_query(query)}.jpg"), vis)
    return rec
#
# def run_on_image(det, clip_model, preprocess, tokenizer, img_path:Path, query:str,
#                  out_dir:Path=None, topk:int=TOPK, fuse_lambda:float=LAMBDA, fuse_thr:float=FUSE_THR,
#                  yolo_conf:float=YOLO_CONF, yolo_iou:float=YOLO_IOU, imgsz:int=IMGZ,
#                  dump_crops=False, dump_stage="postgate"):
#     im_bgr = cv2.imread(str(img_path));  H,W = im_bgr.shape[:2]
#     r = det.predict(source=str(img_path), imgsz=imgsz, conf=yolo_conf, iou=yolo_iou,
#                     verbose=False, save=False)[0]
#     if r.boxes is None or r.boxes.xyxy.numel()==0:
#         return _save_rec(img_path, query, out_dir, im_bgr, [], [], {"n_proposed":0,"n_after_gate":0,"n_after_topk":0,"n_after_comp":0,"n_kept":0})
#
#     boxes_all = r.boxes.xyxy.cpu().numpy().astype(np.float32)
#     yolo_scores_all = r.boxes.conf.cpu().numpy().astype(np.float32)
#     yolo_cls_all = r.boxes.cls.cpu().numpy().astype(int)
#     n_proposed = boxes_all.shape[0]
#
#     route, subtype_target, siblings = match_group_and_subtype(query)
#     allow_ids = route["allow_ids"]; strict = route["strict"]
#     pos_prompts, neg_prompts = route["pos_prompts"], route["neg_prompts"]
#
#     boxes, yolo_scores, yolo_cls = boxes_all, yolo_scores_all, yolo_cls_all
#     if allow_ids is not None:
#         gate = np.isin(yolo_cls, list(allow_ids))
#         boxes, yolo_scores, yolo_cls = boxes[gate], yolo_scores[gate], yolo_cls[gate]
#         if boxes.shape[0]==0 and strict:
#             return _save_rec(img_path, query, out_dir, im_bgr, [], [], {"n_proposed":n_proposed,"n_after_gate":0,"n_after_topk":0,"n_after_comp":0,"n_kept":0})
#     n_after_gate = boxes.shape[0]
#
#     if n_after_gate==0:
#         return _save_rec(img_path, query, out_dir, im_bgr, [], [], {"n_proposed":n_proposed,"n_after_gate":0,"n_after_topk":0,"n_after_comp":0,"n_kept":0})
#
#     # pre-overlay (what we send to CLIP)
#     pre_overlay = draw_boxes(im_bgr, boxes, yolo_scores, label="pre") if out_dir is not None else None
#
#     idx = np.argsort(-yolo_scores)[:topk]
#     boxes = boxes[idx]; yolo_scores = yolo_scores[idx]; n_after_topk = boxes.shape[0]
#
#     image_stem = img_path.stem
#     dump_dir = (out_dir / "debug_crops") if (out_dir and dump_crops) else None
#     do_dump = dump_crops and (dump_stage in ["postgate","posttopk"])
#
#     if (route["cfg"] is not None) and (subtype_target is not None) and ("subtypes" in route["cfg"]):
#         ok_mask, s_t, s_best = subtype_competition(
#             clip_model, preprocess, tokenizer, im_bgr, boxes,
#             route["cfg"]["subtypes"], subtype_target, siblings,
#             dump_crops=do_dump, dump_dir=dump_dir, image_stem=image_stem
#         )
#         s_clip_n = norm01(s_t)    # proxy for ranking after GREEN/RED
#         keep_mask_extra = ok_mask
#         n_after_comp = int(ok_mask.sum())
#     else:
#         s_clip, pos_raw, neg_raw = clip_simple(
#             clip_model, preprocess, tokenizer, im_bgr, boxes,
#             pos_prompts, neg_prompts,
#             dump_crops=do_dump, dump_dir=dump_dir, image_stem=image_stem
#         )
#         s_clip_n = norm01(s_clip)
#         keep_mask_extra = (pos_raw - neg_raw) >= MARGIN
#         n_after_comp = int(keep_mask_extra.sum())
#
#     s_yolo_n = norm01(yolo_scores)
#     fused = fuse_lambda * s_clip_n + (1.0 - fuse_lambda) * s_yolo_n
#     keep = (fused >= fuse_thr) & keep_mask_extra
#
#     boxes_out, scores_out = boxes[keep], fused[keep]
#     order = np.argsort(-scores_out); boxes_out, scores_out = boxes_out[order], scores_out[order]
#     counts = dict(n_proposed=int(n_proposed), n_after_gate=int(n_after_gate),
#                   n_after_topk=int(n_after_topk), n_after_comp=int(n_after_comp),
#                   n_kept=int(len(boxes_out)))
#
#     return _save_rec(img_path, query, out_dir, im_bgr, boxes_out, scores_out, counts, pre_overlay=pre_overlay)

def run_on_image(det, clip_model, preprocess, tokenizer, img_path:Path, query:str,
                 out_dir:Path=None, topk:int=TOPK, fuse_lambda:float=LAMBDA, fuse_thr:float=FUSE_THR,
                 yolo_conf:float=YOLO_CONF, yolo_iou:float=YOLO_IOU, imgsz:int=IMGZ,
                 dump_crops=False, dump_stage="postgate"):
    im_bgr = cv2.imread(str(img_path));  H,W = im_bgr.shape[:2]
    r = det.predict(source=str(img_path), imgsz=imgsz, conf=yolo_conf, iou=yolo_iou,
                    verbose=False, save=False)[0]
    if r.boxes is None or r.boxes.xyxy.numel()==0:
        return _save_rec(img_path, query, out_dir, im_bgr, [], [], {"n_proposed":0,"n_after_gate":0,"n_after_topk":0,"n_after_comp":0,"n_kept":0})

    boxes_all = r.boxes.xyxy.cpu().numpy().astype(np.float32)
    yolo_scores_all = r.boxes.conf.cpu().numpy().astype(np.float32)
    yolo_cls_all = r.boxes.cls.cpu().numpy().astype(int)
    n_proposed = boxes_all.shape[0]

    route, subtype_target, siblings = match_group_and_subtype(query)
    allow_ids = route["allow_ids"]; strict = route["strict"]
    pos_prompts, neg_prompts = route["pos_prompts"], route["neg_prompts"]

    boxes, yolo_scores, yolo_cls = boxes_all, yolo_scores_all, yolo_cls_all
    if allow_ids is not None:
        gate = np.isin(yolo_cls, list(allow_ids))
        boxes, yolo_scores, yolo_cls = boxes[gate], yolo_scores[gate], yolo_cls[gate]
        if boxes.shape[0]==0 and strict:
            return _save_rec(img_path, query, out_dir, im_bgr, [], [], {"n_proposed":n_proposed,"n_after_gate":0,"n_after_topk":0,"n_after_comp":0,"n_kept":0})
    n_after_gate = boxes.shape[0]

    if n_after_gate==0:
        return _save_rec(img_path, query, out_dir, im_bgr, [], [], {"n_proposed":n_proposed,"n_after_gate":0,"n_after_topk":0,"n_after_comp":0,"n_kept":0})

    # pre-overlay (what we send to CLIP)
    pre_overlay = draw_boxes(im_bgr, boxes, yolo_scores, label="pre") if out_dir is not None else None

    idx = np.argsort(-yolo_scores)[:topk]
    boxes = boxes[idx]; yolo_scores = yolo_scores[idx]; n_after_topk = boxes.shape[0]

    image_stem = img_path.stem
    dump_dir = (out_dir / "debug_crops") if (out_dir and dump_crops) else None
    do_dump = dump_crops and (dump_stage in ["postgate","posttopk"])

    if (route["cfg"] is not None) and (subtype_target is not None) and ("subtypes" in route["cfg"]):
        # --- THIS IS THE NEW, CORRECTED BLOCK ---
        ok_mask, s_t, s_best = subtype_competition(
            clip_model, preprocess, tokenizer, im_bgr, boxes,
            route["cfg"]["subtypes"], subtype_target, siblings,
            dump_crops=do_dump, dump_dir=dump_dir, image_stem=image_stem
        )
        if ok_mask.any():
            # preferred path: subtype competition succeeded on at least one box
            s_clip_n = norm01(s_t)      # rank winners by their target score
            keep_mask_extra = ok_mask   # only GREEN pass
            n_after_comp = int(ok_mask.sum())
        else:
            # fallback: competition too strict → use generic pos/neg scoring
            s_clip, pos_raw, neg_raw = clip_simple(
                clip_model, preprocess, tokenizer, im_bgr, boxes,
                pos_prompts, neg_prompts,
                dump_crops=do_dump, dump_dir=dump_dir, image_stem=image_stem
            )
            s_clip_n = norm01(s_clip)
            # keep a light margin to avoid blatant mismatches; you can set this to np.ones_like(s_clip, dtype=bool) to keep everything
            keep_mask_extra = (pos_raw - neg_raw) >= MARGIN
            n_after_comp = int(keep_mask_extra.sum())  # count after fallback
        # --- END OF NEW BLOCK ---
    else:
        s_clip, pos_raw, neg_raw = clip_simple(
            clip_model, preprocess, tokenizer, im_bgr, boxes,
            pos_prompts, neg_prompts,
            dump_crops=do_dump, dump_dir=dump_dir, image_stem=image_stem
        )
        s_clip_n = norm01(s_clip)
        keep_mask_extra = (pos_raw - neg_raw) >= MARGIN
        n_after_comp = int(keep_mask_extra.sum())

    s_yolo_n = norm01(yolo_scores)
    fused = fuse_lambda * s_clip_n + (1.0 - fuse_lambda) * s_yolo_n
    keep = (fused >= fuse_thr) & keep_mask_extra

    boxes_out, scores_out = boxes[keep], fused[keep]
    order = np.argsort(-scores_out); boxes_out, scores_out = boxes_out[order], scores_out[order]
    counts = dict(n_proposed=int(n_proposed), n_after_gate=int(n_after_gate),
                  n_after_topk=int(n_after_topk), n_after_comp=int(n_after_comp),
                  n_kept=int(len(boxes_out)))

    return _save_rec(img_path, query, out_dir, im_bgr, boxes_out, scores_out, counts, pre_overlay=pre_overlay)

def main():
    global TEMP,PERC_Q,NEG_BETA,MARGIN,THR,DELTA,FUSE_THR
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--input", default="bqa/bqa_test")
    ap.add_argument("--query", required=True)
    ap.add_argument("--output", default="bqa/bqa_result")
    ap.add_argument("--clip", default="ViT-B-32")
    ap.add_argument("--clip_pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--fuse_lambda", type=float, default=LAMBDA)
    ap.add_argument("--fuse_thr", type=float, default=FUSE_THR)
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--yolo_conf", type=float, default=YOLO_CONF)
    ap.add_argument("--yolo_iou", type=float, default=YOLO_IOU)
    ap.add_argument("--imgsz", type=int, default=IMGZ)
    # CLIP & competition
    ap.add_argument("--temp", type=float, default=TEMP)
    ap.add_argument("--perc_q", type=int, default=PERC_Q)
    ap.add_argument("--neg_beta", type=float, default=NEG_BETA)
    ap.add_argument("--margin", type=float, default=MARGIN)
    ap.add_argument("--thr", type=float, default=THR)
    ap.add_argument("--delta", type=float, default=DELTA)
    # debugging
    ap.add_argument("--dump_crops", type=int, default=1, help="1=save CLIP crops per image")
    ap.add_argument("--dump_stage", default="postgate", choices=["postgate","posttopk"])
    args = ap.parse_args()

    TEMP=args.temp; PERC_Q=args.perc_q; NEG_BETA=args.neg_beta; MARGIN=args.margin
    THR=args.thr; DELTA=args.delta; FUSE_THR=args.fuse_thr

    det, clip_model, preprocess, tokenizer = load_models(args.weights, args.clip, args.clip_pretrained)
    in_path, out_dir = Path(args.input), Path(args.output)

    imgs = [in_path] if in_path.is_file() else sorted([*in_path.glob("*.jpg"),*in_path.glob("*.jpeg"),*in_path.glob("*.png")])
    out_csv = out_dir / f"bqa_{_clean_query(args.query).replace(' ','_')}.csv"
    rows = []
    for p in tqdm(imgs, desc="BQA"):
        rows.append(run_on_image(det, clip_model, preprocess, tokenizer, p, args.query, out_dir,
                                 args.topk, args.fuse_lambda, args.fuse_thr,
                                 args.yolo_conf, args.yolo_iou, args.imgsz,
                                 dump_crops=bool(args.dump_crops), dump_stage=args.dump_stage))
    try:
        import pandas as pd
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"[OK] Wrote {out_csv}")
    except Exception as e:
        print("[WARN] Could not write CSV:", e)

if __name__ == "__main__":
    main()
