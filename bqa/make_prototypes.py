#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import cv2, torch, open_clip
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm

# same taxonomy you already defined; keep it short for mining
SHARP_SUBTYPES = {
    "knife":    ["knife", "chef knife", "kitchen knife", "elongated single blade"],
    "scissors": ["scissors", "finger holes", "two blades with pivot"],
    "fork":     ["fork", "four prongs", "tines"],
    "pin":      ["pin", "thumbtack", "small round head", "thin needle-like shaft"],
}
MEDICAL_SUBTYPES = {
    "syringe":   ["syringe", "plunger barrel needle", "graduated markings"],
    "test_tube": ["test tube", "glass lab tube", "rounded bottom"],
    "gloves":    ["gloves", "five fingers latex"],
    "mask":      ["surgical mask", "ear loops pleats"],
}

SHARP_NEG = {
    "knife":    ["fork", "scissors", "pin", "needle", "nail", "screw", "wire"],
    "scissors": ["knife", "razor", "box cutter", "tongs", "tweezers"],
    "fork":     ["knife", "spoon", "comb", "prongs short metal tool"],
    "pin":      ["needle", "nail", "screw", "wire", "staple", "toothpick"],
}
MED_NEG = {
    "syringe":   ["test tube", "pipette", "pen", "marker", "dropper"],
    "test_tube": ["syringe", "vial", "pipette", "bottle"],
    "gloves":    ["plastic bag", "cloth", "mask"],
    "mask":      ["gloves", "paper tissue", "cloth"],
}

def norm(x): return x / (x.norm(dim=-1, keepdim=True) + 1e-9)

@torch.no_grad()
def encode_texts(model, tokenizer, prompts, device):
    toks = tokenizer(prompts).to(device)
    t = model.encode_text(toks)
    return norm(t)

@torch.no_grad()
def encode_images(model, preprocess, pil_images, device):
    batch = torch.cat([preprocess(im).unsqueeze(0) for im in pil_images], dim=0).to(device)
    z = model.encode_image(batch)
    return norm(z)

def mine_from_split(det, clip_model, preprocess, tokenizer,
                    image_paths,
                    allow_cls_ids,
                    subtype_prompts,
                    outdir,
                    per_subtype_max=120,
                    margin=0.10,
                    temp=0.05,
                    device="cuda"):

    txt = {st: encode_texts(clip_model, tokenizer, prompts, device) for st, prompts in subtype_prompts.items()}
    name2id = {n:i for i,n in enumerate(['glass','medical','metal','organic','paper','plastic','sharp-object'])}


    # state per subtype
    banks = {st: [] for st in subtype_prompts}
    img_idx = {st: 0 for st in subtype_prompts}

    for p in tqdm(image_paths, desc=f"Mining {outdir.name}"):
        res = det.predict(source=str(p), imgsz=640, conf=0.15, verbose=False, iou=0.5)[0]
        if res.boxes is None or res.boxes.xyxy.numel() == 0:
            continue

        im_bgr = cv2.imread(str(p))
        if im_bgr is None: continue

        gate = np.isin(res.boxes.cls.cpu().numpy(), list(allow_cls_ids))
        if not gate.any():
            continue
        boxes = res.boxes.xyxy[gate]

        # crops
        crops = []
        for (x1,y1,x2,y2) in boxes:
            x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
            crop = im_bgr[y1:y2, x1:x2, ::-1] # BGR->RGB
            crops.append(Image.fromarray(crop))
        if not crops: continue

        # clip score all crops
        z = encode_images(clip_model, preprocess, crops, device) # [N,D]
        scores = {}
        for st, t in txt.items():
            s = (z @ t.T) / temp
            k = max(1, int(0.9 * s.size(1)))
            topk = s.topk(k, dim=1).values.mean(dim=1)   # pos score

            # subtract max negative similarity (contrastive push)
            if allow_cls_ids == {name2id["sharp-object"]}:
                neg_prompts = SHARP_NEG.get(st, [])
            else:
                neg_prompts = MED_NEG.get(st, [])
            if neg_prompts:
                tneg = encode_texts(clip_model, tokenizer, neg_prompts, device)   # [Q,D]
                sneg = (z @ tneg.T) / temp                                        # [N,Q]
                topk = topk - 0.30 * sneg.max(dim=1).values                       # NEG_BETA=0.30

            scores[st] = topk

        # which subtype wins for each crop?
        score_mat = torch.stack(list(scores.values()), dim=1) # [N, C]
        best_scores, best_idx = score_mat.max(dim=1)
        subtype_names = list(scores.keys())
        best_subtypes = [subtype_names[i] for i in best_idx]

        # margin filter: keep only if best is confidently better than second best
        second_best, _ = score_mat.topk(2, dim=1)
        gap = second_best[:,0] - second_best[:,1]
        keep_mask = gap >= max(0.12, margin)   # be a bit stricter for piles

        # add to banks
        for i, keep in enumerate(keep_mask):
            if not keep.item(): continue
            st = best_subtypes[i]
            if len(banks[st]) < per_subtype_max:
                banks[st].append(z[i].cpu().numpy())
                # also save the image crop for inspection
                crop_dir = outdir / f"{st}_crops"
                crop_dir.mkdir(exist_ok=True)
                crops[i].save(crop_dir/f"{img_idx[st]}.png")
                img_idx[st] += 1

    # save final banks
    outdir.mkdir(exist_ok=True)
    for st, embeddings in banks.items():
        if not embeddings: continue
        mat = np.stack(embeddings)
        print(f"  {st}: mined {mat.shape[0]} prototypes")
        np.save(outdir/f"{st}.npy", mat)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--train", required=True, help="path to train/images")
    ap.add_argument("--val", required=True, help="path to valid/images")
    ap.add_argument("--clip", default="ViT-B-32")
    ap.add_argument("--clip_pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--outdir", default="bqa/prototypes")
    ap.add_argument("--group", choices=["sharp-object","medical"], required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    det = YOLO(args.weights)
    model, _, preprocess = open_clip.create_model_and_transforms(args.clip, pretrained=args.clip_pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(args.clip)
    model.eval()

    # map class name -> id
    CLASSES = ['glass','medical','metal','organic','paper','plastic','sharp-object']
    name2id = {n:i for i,n in enumerate(CLASSES)}

    if args.group == "sharp-object":
        allow = {name2id["sharp-object"]}
        prompts = SHARP_SUBTYPES
        out = Path(args.outdir)/"sharp-object"
    else:
        allow = {name2id["medical"]}
        prompts = MEDICAL_SUBTYPES
        out = Path(args.outdir)/"medical"

    train_imgs = sorted(list(Path(args.train).glob("*.jpg")) + list(Path(args.train).glob("*.png")))
    val_imgs = sorted(list(Path(args.val).glob("*.jpg")) + list(Path(args.val).glob("*.png")))
    all_imgs = train_imgs + val_imgs

    mine_from_split(det, model, preprocess, tokenizer, all_imgs, allow, prompts, out, device=device)

if __name__ == "__main__":
    main()
