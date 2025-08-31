# #!/usr/bin/env python

# !/usr/bin/env python
# train_hazard.py

import torch
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.models.yolo.detect.train_hazard import HazardDetectionTrainer

from pathlib import Path

# === FIX: Register custom arguments before any functions are called ===
# This makes medical_idx and lambda_hz known to the configuration validator.
DEFAULT_CFG.medical_idx = 1
DEFAULT_CFG.lambda_hz = 0.5


# === END OF FIX ===


def train_stage1():
    """Stage 1: Train hazard head with frozen backbone."""
    print("=" * 50)
    print("Stage 1: Training hazard head with frozen backbone")
    print("=" * 50)

    args = get_cfg(DEFAULT_CFG)

    # === FIX: Correctly load pretrained weights ===
    # Pass the .pt file path to the 'model' argument. The YAML path will
    # be automatically inferred from the model architecture.
    pretrained_path = 'weights/yolov8m-baseline-640.pt'
    if Path(pretrained_path).exists():
        args.model = pretrained_path
        print(f"Loading pretrained weights from {pretrained_path}")
    else:
        # If no pretrained weights, start from the YAML definition
        args.model = 'ultralytics/cfg/models/v8/yolov8m-hazard.yaml'
        print(f"Pretrained weights not found. Training from YAML definition.")
    # === END OF FIX ===

    args.data = 'dataset-640/data.yaml'
    args.epochs = 5
    args.batch = 8
    args.imgsz = 640
    args.project = 'runs/hazard'
    args.name = 'stage1_frozen'
    args.medical_idx = 1
    args.lambda_hz = 0.5
    args.device = 0 if torch.cuda.is_available() else 'cpu'

    # Freeze backbone (first 10 layers)
    args.freeze = 10
    args.lr0 = 1e-3

    # === FIX: Use vars(args) to pass arguments as a dictionary ===
    trainer = HazardDetectionTrainer(overrides=vars(args))
    trainer.train()

    return trainer.best


def train_stage2(checkpoint_path):
    """Stage 2: Fine-tune entire network."""
    print("=" * 50)
    print("Stage 2: Fine-tuning entire network")
    print("=" * 50)

    args = get_cfg(DEFAULT_CFG)
    args.model = checkpoint_path  # Use checkpoint from stage 1
    args.data = 'dataset-640/data.yaml'
    args.epochs = 10
    args.batch = 4
    args.imgsz = 640
    args.project = 'runs/hazard'
    args.name = 'stage2_finetuned'
    args.medical_idx = 1
    args.lambda_hz = 0.5
    args.device = 0 if torch.cuda.is_available() else 'cpu'

    # No freezing, lower learning rate
    args.freeze = None
    args.lr0 = 1e-4
    args.resume = True

    # === FIX: Use vars(args) for consistency ===
    trainer = HazardDetectionTrainer(overrides=vars(args))
    trainer.train()

    return trainer.best


if __name__ == '__main__':
    # Stage 1
    stage1_best = train_stage1()
    print(f"\nStage 1 complete! Best model: {stage1_best}")

    # Stage 2
    stage2_best = train_stage2(stage1_best)
    print(f"\nStage 2 complete! Best model: {stage2_best}")

    # Validate final model
    from ultralytics import YOLO

    model = YOLO(stage2_best)
    results = model.val()

    print("\n" + "=" * 50)
    print("Final Validation Results:")
    print("=" * 50)
    print(f"Overall mAP50: {results.box.map50:.4f}")
    print(f"Overall mAP50-95: {results.box.map:.4f}")


#!/usr/bin/env python
# train_hazard.py
#
# import torch
# from ultralytics.cfg import get_cfg
# from ultralytics.utils import DEFAULT_CFG
# from ultralytics.models.yolo.detect.train_hazard import HazardDetectionTrainer
# from pathlib import Path
#
#
# def train_stage1():
#     """Stage 1: Train hazard head with frozen backbone."""
#     print("=" * 50)
#     print("Stage 1: Training hazard head with frozen backbone")
#     print("=" * 50)
#
#     args = get_cfg(DEFAULT_CFG)
#     args.model = 'ultralytics/cfg/models/v8/yolov8m-hazard.yaml'
#     args.data = 'dataset-640/data.yaml'  # Corrected path
#     args.epochs = 5
#     args.batch = 8
#     args.imgsz = 640
#     args.project = 'runs/hazard'
#     args.name = 'stage1_frozen'
#     args.medical_idx = 1
#     args.lambda_hz = 0.5
#     args.device = 0 if torch.cuda.is_available() else 'cpu'
#
#     # Freeze backbone (first 10 layers)
#     args.freeze = 10
#     args.lr0 = 1e-3
#
#     # Load pretrained weights if available
#     pretrained_path = 'weights/yolov8m-baseline-640.pt'  # Corrected path
#     if Path(pretrained_path).exists():
#         args.weights = pretrained_path
#         print(f"Loading pretrained weights from {pretrained_path}")
#     else:
#         print(f"Pretrained weights not found at {pretrained_path}. Training from scratch.")
#
#     trainer = HazardDetectionTrainer(overrides=dict(args))
#     trainer.train()
#
#     return trainer.best
#
#
# def train_stage2(checkpoint_path):
#     """Stage 2: Fine-tune entire network."""
#     print("=" * 50)
#     print("Stage 2: Fine-tuning entire network")
#     print("=" * 50)
#
#     args = get_cfg(DEFAULT_CFG)
#     args.model = checkpoint_path  # Use checkpoint from stage 1
#     args.data = 'dataset-640/data.yaml'  # Corrected path
#     args.epochs = 10
#     args.batch = 4
#     args.imgsz = 640
#     args.project = 'runs/hazard'
#     args.name = 'stage2_finetuned'
#     args.medical_idx = 1
#     args.lambda_hz = 0.5
#     args.device = 0 if torch.cuda.is_available() else 'cpu'
#
#     # No freezing, lower learning rate
#     args.freeze = None
#     args.lr0 = 1e-4
#     args.resume = True
#
#     trainer = HazardDetectionTrainer(overrides=dict(args))
#     trainer.train()
#
#     return trainer.best
#
#
# if __name__ == '__main__':
#     # Stage 1
#     stage1_best = train_stage1()
#     print(f"\nStage 1 complete! Best model: {stage1_best}")
#
#     # Stage 2
#     stage2_best = train_stage2(stage1_best)
#     print(f"\nStage 2 complete! Best model: {stage2_best}")
#
#     # Validate final model
#     from ultralytics import YOLO
#
#     model = YOLO(stage2_best)
#     results = model.val()
#
#     print("\n" + "=" * 50)
#     print("Final Validation Results:")
#     print("=" * 50)
#     print(f"Overall mAP50: {results.box.map50:.4f}")
#     print(f"Overall mAP50-95: {results.box.map:.4f}")


# # train_hazard.py
#
# import torch
# from ultralytics.cfg import get_cfg
# from ultralytics.utils import DEFAULT_CFG
# from ultralytics.models.yolo.detect.train_hazard import HazardDetectionTrainer
#
#
# def train_stage1():
#     """Stage 1: Train hazard head with frozen backbone."""
#     print("=" * 50)
#     print("Stage 1: Training hazard head with frozen backbone")
#     print("=" * 50)
#
#     args = get_cfg(DEFAULT_CFG)
#     args.model = 'ultralytics/cfg/models/v8/yolov8m-hazard.yaml'
#     args.data = 'dataset-640/data.yaml'  # Update to your data path
#     args.epochs = 5
#     args.batch = 8
#     args.imgsz = 640
#     args.project = 'runs/hazard'
#     args.name = 'stage1_frozen'
#     args.medical_idx = 1  # Your medical class index
#     args.lambda_hz = 0.5
#     args.device = 0 if torch.cuda.is_available() else 'cpu'
#
#     # Freeze backbone (first 10 layers)
#     args.freeze = 10
#     args.lr0 = 1e-3
#
#     # Load pretrained weights if available
#     pretrained_path = '../weights/best.pt'
#     if Path(pretrained_path).exists():
#         args.weights = pretrained_path
#         print(f"Loading pretrained weights from {pretrained_path}")
#
#     trainer = HazardDetectionTrainer(overrides=dict(args))
#     trainer.train()
#
#     return trainer.best
#
#
# def train_stage2(checkpoint_path):
#     """Stage 2: Fine-tune entire network."""
#     print("=" * 50)
#     print("Stage 2: Fine-tuning entire network")
#     print("=" * 50)
#
#     args = get_cfg(DEFAULT_CFG)
#     args.model = checkpoint_path  # Use checkpoint from stage 1
#     args.data = '../datasets/data.yaml'  # Update to your data path
#     args.epochs = 10
#     args.batch = 4
#     args.imgsz = 640
#     args.project = 'runs/hazard'
#     args.name = 'stage2_finetuned'
#     args.medical_idx = 1
#     args.lambda_hz = 0.5
#     args.device = 0 if torch.cuda.is_available() else 'cpu'
#
#     # No freezing, lower learning rate
#     args.freeze = None
#     args.lr0 = 1e-4
#     args.resume = True
#
#     trainer = HazardDetectionTrainer(overrides=dict(args))
#     trainer.train()
#
#     return trainer.best
#
#
# if __name__ == '__main__':
#     from pathlib import Path
#
#     # Stage 1
#     stage1_best = train_stage1()
#     print(f"\nStage 1 complete! Best model: {stage1_best}")
#
#     # Stage 2
#     stage2_best = train_stage2(stage1_best)
#     print(f"\nStage 2 complete! Best model: {stage2_best}")
#
#     # Validate final model
#     from ultralytics import YOLO
#
#     model = YOLO(stage2_best)
#     results = model.val()
#
#     print("\n" + "=" * 50)
#     print("Final Validation Results:")
#     print("=" * 50)
#     print(f"Overall mAP50: {results.box.map50:.4f}")
#     print(f"Overall mAP50-95: {results.box.map:.4f}")