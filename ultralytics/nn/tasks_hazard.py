# ultralytics/nn/tasks_hazard.py
import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.hazard_loss import HazardLoss


class HazardDetectionModel(DetectionModel):
    """YOLOv8 detection model with HazardDetect head."""

    def __init__(self, cfg='yolov8m-hazard.yaml', ch=3, nc=None, verbose=True):
        """Initialize model with HazardDetect head."""
        super().__init__(cfg, ch, nc, verbose)
        # The model already has HazardDetect as the last layer from YAML

    def init_criterion(self):
        """Initialize the loss criterion with hazard loss."""
        return HazardDetectionLoss(self)


class HazardDetectionLoss(v8DetectionLoss):
    """Detection loss that handles HazardDetect head outputs."""

    def __init__(self, model):
        """Initialize with hazard loss component."""
        super().__init__(model)

        # Get hazard parameters from the HazardDetect head
        detect_head = model.model[-1]  # Last layer should be HazardDetect
        if hasattr(detect_head, 'medical_idx'):
            self.hazard_criterion = HazardLoss(
                medical_idx=detect_head.medical_idx,
                pos_weight=2.0
            )
            self.lambda_hz = detect_head.lambda_hz
        else:
            self.hazard_criterion = None
            self.lambda_hz = 0.0

    def __call__(self, preds, batch):
        """Calculate combined detection and hazard loss."""
        # Handle HazardDetect output format
        hazard_logits = None
        if isinstance(preds, tuple) and len(preds) == 2:
            # HazardDetect returns (detection_preds, hazard_logits) during training
            preds, hazard_logits = preds

        # Calculate standard detection loss
        det_loss, loss_items = super().__call__(preds, batch)

        # Add hazard loss if available
        if hazard_logits is not None and self.hazard_criterion is not None:
            hz_loss = self.hazard_criterion(hazard_logits, batch, self.device)

            # Add weighted hazard loss to total
            total_loss = det_loss + self.lambda_hz * hz_loss

            # Extend loss items for logging (box, cls, dfl, hazard)
            loss_items = torch.cat([loss_items, hz_loss.detach().unsqueeze(0)])

            return total_loss, loss_items

        return det_loss, loss_items