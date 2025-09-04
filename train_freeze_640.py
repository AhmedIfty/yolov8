# scripts/train_freeze_640.py
from ultralytics import YOLO

# DATA_YAML = 'dataset/waste_cleanval.yaml'  # or 'dataset/data.yaml' if you skipped clean-val

DATA_YAML = 'dataset-640/data.yaml'

if __name__ == '__main__':
    model = YOLO('yolov8m.pt')
    model.train(
        data=DATA_YAML,
        epochs=15,               # short freeze probe
        imgsz=640,
        batch=16,                # drop to 12/8 if OOM
        multi_scale=True,
        mosaic=1.0, mixup=0.2,  # heavy aug; reduce later if unstable
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10, translate=0.10, scale=0.50, shear=2.0, perspective=0.0005,
        fliplr=0.5, flipud=0.0,
        optimizer='SGD', lr0=0.01, lrf=0.1, cos_lr=False,
        momentum=0.90, weight_decay=0.0005,
        warmup_epochs=3,
        freeze=10,               # freeze early backbone layers
        patience=20,             # early stop if it stalls
        amp=True, workers=8,
        device=0,
        project='runs/train', name='waste_freeze_640', exist_ok=True
    )