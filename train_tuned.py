# scripts/train_640_single.py
from ultralytics import YOLO
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="yolov8m.pt")
    p.add_argument("--data", default="dataset-refined-v4/data.yaml")
    p.add_argument("--epochs", type=int, default=50)       # solid single-run target (early stop = 20)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=0)
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default="refined-exp4-yolov8m")
    return p.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.weights)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        # Augmentations (Stage-B style: moderate & stable at 640)
        multi_scale=False,
        mosaic=0.5,
        mixup=0.1,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10, translate=0.10, scale=0.50, shear=2.0, perspective=0.0005,
        fliplr=0.5, flipud=0.0,

        # Optimizer & schedule (your params)
        optimizer="SGD", lr0=0.01, lrf=0.1, cos_lr=False,
        momentum=0.90, weight_decay=0.0005,
        warmup_epochs=3,

        # Practicalities
        amp=True,
        workers=1,        # Windows-friendly; raise if stable
        cache=True,       # speed-up; set False if RAM/pagefile complains
        patience=20,      # early stop
        device=args.device,
        project=args.project, name=args.name, exist_ok=True,
    )

    # (Optional) quick test-split eval at the end so you see final numbers
    #model.val(data=args.data, split="test", imgsz=args.imgsz, device=args.device, verbose=False)

if __name__ == "__main__":
    main()
