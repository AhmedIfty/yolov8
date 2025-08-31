# test_train.py

import torch
from pathlib import Path
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.models.yolo.detect.train_hazard import HazardDetectionTrainer

# === START OF THE FIX ===
# Register your custom arguments with the default configuration.
# This makes them valid arguments that the trainer can accept.
DEFAULT_CFG.medical_idx = 1  # Default value for medical class index
DEFAULT_CFG.lambda_hz = 0.5  # Default value for hazard loss weight
# === END OF THE FIX ===


def test_training():
    """Test that training can start without errors."""

    print("Testing training initialization...")

    # Setup minimal config
    args = get_cfg(DEFAULT_CFG)
    args.model = 'ultralytics/cfg/models/v8/yolov8m-hazard.yaml'
    args.data = 'coco8.yaml'
    args.epochs = 1
    args.batch = 2
    args.imgsz = 320
    args.device = 'cpu'
    # You are now overriding the default values you just registered
    args.medical_idx = 1
    args.lambda_hz = 0.5
    args.project = 'test_runs'
    args.name = 'hazard_test'
    args.exist_ok = True
    args.verbose = True

    try:
        # Initialize trainer - this will now succeed
        trainer = HazardDetectionTrainer(overrides=vars(args))
        print("✓ Trainer initialized successfully")

        # ... (rest of your test script remains the same)

    except Exception as e:
        print(f"\n❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_training()
    assert success, "Training test failed"


# import torch
# from pathlib import Path
# from ultralytics.cfg import get_cfg
# from ultralytics.utils import DEFAULT_CFG
# from ultralytics.models.yolo.detect.train_hazard import HazardDetectionTrainer
#
#
# def test_training():
#     """Test that training can start without errors."""
#
#     print("Testing training initialization...")
#
#     # Setup minimal config
#     args = get_cfg(DEFAULT_CFG)
#     args.model = 'ultralytics/cfg/models/v8/yolov8m-hazard.yaml'
#     args.data = 'coco8.yaml'  # Use COCO8 for testing (small dataset)
#     args.epochs = 1  # Just test 1 epoch
#     args.batch = 2
#     args.imgsz = 320  # Smaller size for testing
#     args.device = 'cpu'  # Use CPU for testing
#     args.medical_idx = 1
#     args.lambda_hz = 0.5
#     args.project = 'test_runs'
#     args.name = 'hazard_test'
#     args.exist_ok = True
#     args.verbose = True
#
#     try:
#         # Initialize trainer
#         trainer = HazardDetectionTrainer(overrides=dict(args))
#         print("✓ Trainer initialized successfully")
#
#         # Get model (this will build the model)
#         model = trainer.get_model()
#         print(f"✓ Model created: {model.model[-1].__class__.__name__}")
#
#         # Verify hazard head exists
#         assert hasattr(model.model[-1], 'hazard_head'), "Model should have hazard_head"
#         print("✓ Hazard head found in model")
#
#         # Test a single forward pass with dummy data
#         dummy_img = torch.randn(2, 3, 320, 320)  # batch_size=2
#         model.train()
#         output = model.predict(dummy_img)
#
#         if isinstance(output, tuple) and len(output) == 2:
#             print("✓ Model forward pass successful (returns detection and hazard outputs)")
#
#         print("\n✅ Training test passed! Model is ready for full training.")
#         return True
#
#     except Exception as e:
#         print(f"\n❌ Training test failed: {e}")
#         import traceback
#         traceback.print_exc()
#         return False
#
#
# if __name__ == "__main__":
#     success = test_training()
#     assert success, "Training test failed"