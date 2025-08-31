# test_hazard.py
import sys
import torch
from pathlib import Path


def test_hazard_integration():
    """Test the complete hazard head integration."""

    try:
        # Test 1: Import HazardDetect
        print("Test 1: Importing HazardDetect...")
        from ultralytics.nn.modules.head import HazardDetect
        head = HazardDetect(nc=7, ch=(256, 512, 768))
        print(f"✓ HazardDetect created with hazard_head: {hasattr(head, 'hazard_head')}")

        # Test 2: Load HazardDetectionModel
        print("\nTest 2: Loading HazardDetectionModel...")
        from ultralytics.nn.tasks_hazard import HazardDetectionModel

        config_path = Path('ultralytics/cfg/models/v8/yolov8m-hazard.yaml')
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")

        model = HazardDetectionModel(
            cfg=str(config_path),
            nc=7,
            verbose=True  # Set to True to see model architecture
        )
        print(f"✓ Model created successfully")
        print(f"✓ Last layer type: {model.model[-1].__class__.__name__}")

        # Verify it's HazardDetect
        assert model.model[-1].__class__.__name__ == "HazardDetect", "Last layer should be HazardDetect"

        # Test 3: Forward pass
        print("\nTest 3: Testing forward pass...")
        dummy_input = torch.randn(1, 3, 640, 640)

        # Test eval mode (inference)
        model.eval()
        with torch.no_grad():
            # output = model.predict(dummy_input)
            output = model(dummy_input)
            print(f"✓ Inference mode: output shape = {output.shape if hasattr(output, 'shape') else 'tuple/list'}")

        # Test train mode
        model.train()
        # output = model.predict(dummy_input)
        output = model(dummy_input)
        if isinstance(output, tuple) and len(output) == 2:
            det_output, hz_output = output
            print(f"✓ Training mode: detection output = list of {len(det_output)} tensors")
            print(f"✓ Training mode: hazard output shape = {hz_output.shape}")

        print("\n✅ All tests passed! Ready to train.")
        return True

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_hazard_integration()
    assert success, "Tests failed"