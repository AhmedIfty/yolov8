from ultralytics import YOLO
import argparse
import torch
import random
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    # Use the CBAM yaml as the model definition
    p.add_argument("--model",   default="yolov8m-cbam.yaml")
    # Load COCO-pretrained YOLOv8m weights for partial transfer
    p.add_argument("--weights", default="yolov8m.pt")
    p.add_argument("--data",    default="dataset-refined-v4/data.yaml")
    p.add_argument("--epochs",  type=int, default=50)
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--batch",   type=int, default=8)
    p.add_argument("--device",  default=0)
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name",    default="refined-exp4-yolov8m-cbam")
    p.add_argument("--seed",    type=int, default=0)    # reproducibility helper
    return p.parse_args()

def set_seed(seed):
    if seed is None:
        return
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # keep False for speed
    torch.backends.cudnn.benchmark = True

def main():
    args = parse_args()
    set_seed(args.seed)

    # Build model graph from YAML with our CBAM layers
    model = YOLO(args.model)

    # Load COCO pretrained weights (partial load):
    # - shared layers are loaded
    # - new CBAM params are left randomly init and will train from scratch
    model.load(args.weights)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,

        # your established “stable@640” augs/schedule
        multi_scale=False,
        mosaic=0.5, mixup=0.1,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10, translate=0.10, scale=0.50, shear=2.0, perspective=0.0005,
        fliplr=0.5, flipud=0.0,

        optimizer="SGD", lr0=0.01, lrf=0.1, cos_lr=False,
        momentum=0.90, weight_decay=0.0005,
        warmup_epochs=3,

        amp=True,
        workers=1,
        cache=True,
        patience=20,
        device=args.device,
        project=args.project, name=args.name, exist_ok=True,
    )

    # (Optional) quick test-split eval from script
    # model.val(data=args.data, split="test", imgsz=args.imgsz, device=args.device, conf=0.25, iou=0.45)

if __name__ == "__main__":
    main()
