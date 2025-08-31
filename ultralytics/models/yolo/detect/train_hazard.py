# ultralytics/models/yolo/detect/train_hazard.py
from pathlib import Path
from ultralytics.models.yolo.detect.train import DetectionTrainer

from ultralytics.nn.tasks_hazard import HazardDetectionModel
from ultralytics.utils import DEFAULT_CFG, RANK

import copy
from ultralytics.models.yolo.detect import DetectionValidator
# Import your new custom validator class
from ultralytics.models.yolo.detect.val_hazard import HazardDetectionValidator

class HazardDetectionTrainer(DetectionTrainer):
    """Trainer for YOLOv8 with HazardDetect head."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize trainer with custom loss names."""
        super().__init__(cfg, overrides, _callbacks)
        # Add hazard loss to loss names for logging
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss", "hz_loss")

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Return a HazardDetectionModel, always building from the custom YAML
        and then loading weights if they are provided.
        """
        # 1. ALWAYS use your custom YAML to define the model architecture.
        #    This ensures the HazardDetect head is always part of the model structure,
        #    which is crucial for loading weights correctly.
        hazard_yaml_path = 'ultralytics/cfg/models/v8/yolov8m-hazard.yaml'

        # 2. Build the model with the correct architecture from your custom YAML.
        #    The 'nc' (number of classes) is taken from the dataset info.
        model = HazardDetectionModel(hazard_yaml_path, nc=self.data["nc"], verbose=verbose and RANK == -1)

        # 3. Load the weights into the correctly structured model.
        #    The `weights` argument will contain the path to your '.pt' file.
        if weights:
            # Check if the weights file path is a string before loading
            if isinstance(weights, str):
                model.load(weights)
            else:
                # Handle cases where weights might be a state_dict or other object,
                # though this is less common in typical training scripts.
                print("Warning: Skipping loading weights. Expected a file path string.")

        return model

    def get_validator(self):
        """
        Returns an instance of the custom HazardDetectionValidator.
        """
        val_args = copy.deepcopy(self.args)

        # Clean the arguments for the validator
        if hasattr(val_args, 'medical_idx'):
            del val_args.medical_idx
        if hasattr(val_args, 'lambda_hz'):
            del val_args.lambda_hz

        # Return an instance of your NEW custom validator
        return HazardDetectionValidator(self.test_loader, save_dir=self.save_dir, args=val_args)

    # def get_validator(self):
    #     """Return standard validator but strip hazard output."""
    #     val_args = copy.deepcopy(self.args)
    #     if hasattr(val_args, "medical_idx"):
    #         del val_args.medical_idx
    #     if hasattr(val_args, "lambda_hz"):
    #         del val_args.lambda_hz
    #
    #     # Wrap DetectionValidator to strip hazard logits
    #     base_validator = DetectionValidator(self.test_loader, save_dir=self.save_dir, args=val_args)
    #
    #     old_postprocess = base_validator.postprocess
    #
    #     def wrapped_postprocess(preds):
    #         # if preds is tuple/list => first element is detection output
    #         if isinstance(preds, (tuple, list)):
    #             preds = preds[0]
    #         return old_postprocess(preds)
    #
    #     base_validator.postprocess = wrapped_postprocess
    #     return base_validator

    # === END OF THE FIX ===

    # def get_model(self, cfg=None, weights=None, verbose=True):
    #     """Return a HazardDetectionModel with HazardDetect head."""
    #     # Use the yolov8m-hazard.yaml config that includes HazardDetect
    #     if cfg is None:
    #         cfg = Path('ultralytics/cfg/models/v8/yolov8m-hazard.yaml')
    #         if not cfg.exists():
    #             cfg = 'yolov8m.yaml'  # Fallback to standard config
    #
    #     model = HazardDetectionModel(str(cfg), nc=self.data["nc"], verbose=verbose and RANK == -1)
    #
    #     # Set hazard parameters if provided in args
    #     detect_head = model.model[-1]
    #     if hasattr(detect_head, 'medical_idx'):
    #         if hasattr(self.args, 'medical_idx'):
    #             detect_head.medical_idx = self.args.medical_idx
    #         if hasattr(self.args, 'lambda_hz'):
    #             detect_head.lambda_hz = self.args.lambda_hz
    #
    #     if weights:
    #         model.load(weights)
    #
    #     return model

    def label_loss_items(self, loss_items=None, prefix="train"):
        """Label loss items including hazard loss."""
        keys = [f"{prefix}/{x}" for x in self.loss_names]

        # Case 1: return just the keys (used for metric_keys concatenation)
        if loss_items is None:
            return keys if prefix == "val" else {k: 0.0 for k in keys}

        # Case 2: return actual values during training
        loss_items = loss_items.detach().cpu().numpy() if hasattr(loss_items, 'detach') else loss_items

        if len(loss_items) == 3:
            loss_items = list(loss_items) + [0.0]
        elif len(loss_items) > 4:
            loss_items = loss_items[:4]

        return dict(zip(keys, loss_items))

    # def label_loss_items(self, loss_items=None, prefix="train"):
    #     """Label loss items including hazard loss."""
    #     if loss_items is None:
    #         return {"loss": 0}
    #
    #     loss_items = loss_items.detach().cpu().numpy() if hasattr(loss_items, 'detach') else loss_items
    #
    #     # Ensure we have 4 items (box, cls, dfl, hazard)
    #     if len(loss_items) == 3:
    #         # No hazard loss, add 0
    #         loss_items = list(loss_items) + [0.0]
    #     elif len(loss_items) > 4:
    #         # Take only first 4
    #         loss_items = loss_items[:4]
    #
    #     keys = [f"{prefix}/{x}" for x in self.loss_names]
    #     return dict(zip(keys, loss_items))



#
# class HazardDetectionTrainer(DetectionTrainer):
#     """Trainer for YOLOv8 with HazardDetect head."""
#
#     def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
#         """Initialize trainer with custom loss names."""
#         super().__init__(cfg, overrides, _callbacks)
#         # Add hazard loss to loss names for logging
#         self.loss_names = ("box_loss", "cls_loss", "dfl_loss", "hz_loss")
#
#     def get_model(self, cfg=None, weights=None, verbose=True):
#         """Return a HazardDetectionModel with HazardDetect head."""
#         # Use the yolov8m-hazard.yaml config that includes HazardDetect
#         if cfg is None:
#             cfg = 'ultralytics/cfg/models/v8/yolov8m-hazard.yaml'
#
#         model = HazardDetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
#
#         # Set hazard parameters if provided in args
#         detect_head = model.model[-1]
#         if hasattr(detect_head, 'medical_idx'):
#             if hasattr(self.args, 'medical_idx'):
#                 detect_head.medical_idx = self.args.medical_idx
#             if hasattr(self.args, 'lambda_hz'):
#                 detect_head.lambda_hz = self.args.lambda_hz
#
#         if weights:
#             model.load(weights)
#
#         return model
#
#     def label_loss_items(self, loss_items=None, prefix="train"):
#         """Label loss items including hazard loss."""
#         if loss_items is None:
#             return {"loss": 0}
#
#         loss_items = loss_items.detach().cpu().numpy() if hasattr(loss_items, 'detach') else loss_items
#
#         # Ensure we have 4 items (box, cls, dfl, hazard)
#         if len(loss_items) == 3:
#             # No hazard loss, add 0
#             loss_items = list(loss_items) + [0.0]
#         elif len(loss_items) > 4:
#             # Take only first 4
#             loss_items = loss_items[:4]
#
#         keys = [f"{prefix}/{x}" for x in self.loss_names]
#         return dict(zip(keys, loss_items))