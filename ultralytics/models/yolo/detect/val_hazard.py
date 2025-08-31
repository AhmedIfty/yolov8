# In ultralytics/models/yolo/detect/val_hazard.py

from ultralytics.models.yolo.detect.val import DetectionValidator

class HazardDetectionValidator(DetectionValidator):
    """
    A custom validator for the HazardDetectionModel.

    This validator is designed to correctly handle the dual output (detection predictions
    and hazard logits) from the custom model during the validation phase. It
    overrides the `get_preds` method to ensure only the detection predictions are
    passed to the metrics calculation, avoiding errors in the loss function.
    """

    def get_preds(self):
        """
        Overrides the base `get_preds` method. It runs the model's forward pass
        and then extracts only the detection predictions from the model's tuple output.
        """
        # This calls the forward pass and populates `self.preds` with the raw tuple output
        super().get_preds()

        # `self.preds` is now a list of tuples, e.g., [(det_preds_1, hz_logits_1), ...].
        # We strip out the hazard logits, keeping only the detection predictions.
        self.preds = [pred[0] for pred in self.preds]