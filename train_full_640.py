# scripts/train_full_640_fast.py
from ultralytics import YOLO

DATA_YAML = 'dataset-refined-640-v1/data.yaml'   # use the SAME val split as Stage-A for apples-to-apples

COMMON = dict(
    imgsz=640,
    batch=8,                 # drop to 12/8 if OOM
    multi_scale=False,        # SPEED: keep a fixed 640
    mosaic=0.5,               # lighter than 1.0 for Stage-B
    mixup=0.1,                # lighter than 0.2 for Stage-B
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=10, translate=0.10, scale=0.50, shear=2.0, perspective=0.0005,
    fliplr=0.5, flipud=0.0,
    optimizer='SGD', lr0=0.01, lrf=0.1, cos_lr=False,
    momentum=0.90, weight_decay=0.0005,
    warmup_epochs=3,
    amp=True,
    workers=1,                # Windows-friendly
    cache=True,               # SPEED: cache images/labels in RAM (epoch 2+ faster)
    patience=20,              # early stop
    # Optional pure speed:
    # val=False,             # skip per-epoch val; run model.val() yourself every N epochs
    # plots=False,           # reduce overhead from plotting
)

if __name__ == '__main__':
    # Start from Stage-A best
    model = YOLO('runs/train/refined-exp1-baseline/weights/best.pt')

    model.train(
        data=DATA_YAML,
        epochs=40,            # 15 + 85 ≈ 100 total; early stopping enabled
        freeze=0,
        device=0,
        project='runs/train',
        name='refined-exp1-baseline-full',
        exist_ok=True,
        **COMMON
    )
