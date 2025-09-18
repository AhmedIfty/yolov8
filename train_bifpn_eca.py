# # train_bifpn_eca.py

# train_bifpn_eca.py
# One-stage training for YOLOv8 P2 + PAN + BiFPN_ECA (refiner)
# - Loads COCO weights (yolov8m.pt) for backbone/PAN
# - Trains end-to-end in a single run (no freezes, no swaps)
# - Conservative, stable hyperparams for 640x640

from ultralytics import YOLO
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    # Model & weights
    p.add_argument("--model", default="yolov8m-p2-bifpn-eca.yaml")
    p.add_argument("--weights", default="yolov8m.pt")  # COCO-pretrained init (matching layers only)

    # Data & run setup
    p.add_argument("--data", default="dataset-refined-v3/data.yaml")
    p.add_argument("--epochs", type=int, default=50)  # a bit longer for the new neck
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=0)
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default="refined-exp6_p2-bifpn-eca")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.model)
    model.load(args.weights)

    # One-stage, steady training schedule with all new parameters
    model.train(
        # Data and run setup from argparse
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name="refined-exp_bifpn-final",  # <-- Changed name for the new experiment

        # Optimizer & Schedule
        optimizer="AdamW",          # <-- Confirmed
        lr0=0.001,                  # <-- Confirmed
        weight_decay=0.02,          # <-- Added
        warmup_epochs=5,            # <-- Added
        cos_lr=False,               # <-- Confirmed, ensures flat decay

        # Augmentations (parameters are all confirmed from the new guideline)
        mosaic=0.30,
        mixup=0.05,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10, translate=0.10, scale=0.50, shear=2.0, perspective=0.0005,
        fliplr=0.5, flipud=0.0,

        # Practicalities
        amp=True,                   # <-- Confirmed (enables mixed-precision training)
        workers=1,                  # <-- Added
        cache=True,                 # <-- Added
        patience=25,                # <-- Added (for early stopping)
        exist_ok=True,              # Good practice to keep
        verbose=True,               # Good practice to keep
    )

    # Optional: quick evaluation on test split at the end
    # model.val(data=args.data, split="test", imgsz=args.imgsz, device=args.device, verbose=False)

if __name__ == "__main__":
    main()



# from ultralytics import YOLO
# import argparse
#
# def parse_args():
#     p = argparse.ArgumentParser()
#     # Model & weights
#     p.add_argument("--model", default="yolov8m-p2-bifpn-eca.yaml")
#     p.add_argument("--weights", default="yolov8m.pt")  # COCO-pretrained backbone/head init
#
#     # Data & run setup
#     p.add_argument("--data", default="dataset-refined-v3/data.yaml")
#     p.add_argument("--epochs", type=int, default=50)    # give the new neck a bit more runway
#     p.add_argument("--imgsz", type=int, default=640)
#     p.add_argument("--batch", type=int, default=8)
#     p.add_argument("--device", default=0)
#     p.add_argument("--project", default="runs/train")
#     p.add_argument("--name", default="refined-exp5_p2-bifpn-eca")
#     return p.parse_args()
#
# def main():
#     args = parse_args()
#
#     # Build model from YAML
#     model = YOLO(args.model)
#
#     # Load partial weights from yolov8m (backbone/parts of head). BiFPN params are new -> randomly init.
#     model.load(args.weights)
#
#     # Train end-to-end (no mid-run module swaps)
#     model.train(
#         data=args.data,
#         epochs=args.epochs,
#         imgsz=args.imgsz,
#         batch=args.batch,
#
#         # Augmentations (your stable 640 policy)
#         multi_scale=False,
#         mosaic=0.5,
#         mixup=0.1,
#         hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
#         degrees=10, translate=0.10, scale=0.50, shear=2.0, perspective=0.0005,
#         fliplr=0.5, flipud=0.0,
#
#         # Optimizer & schedule
#         optimizer="SGD", lr0=0.01, lrf=0.1, cos_lr=False,
#         momentum=0.90, weight_decay=0.0005,
#         warmup_epochs=3,
#
#         # Practicalities
#         amp=True,
#         workers=1,            # raise if your env is stable
#         cache=True,           # speed-up if RAM allows
#         patience=20,          # early stop
#         device=args.device,
#         project=args.project, name=args.name, exist_ok=True,
#     )
#
#     # Optional: quick test-split eval
#     # model.val(data=args.data, split="test", imgsz=args.imgsz, device=args.device, verbose=False)
#
# if __name__ == "__main__":
#     main()
