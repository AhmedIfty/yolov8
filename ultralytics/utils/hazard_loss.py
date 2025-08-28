# ultralytics/utils/hazard_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class HazardLoss(nn.Module):
    """Loss function for hazard detection head."""

    def __init__(self, medical_idx=1, pos_weight=2.0):
        super().__init__()
        self.medical_idx = medical_idx
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, hazard_logits, batch_labels, device='cuda'):
        """
        Calculate hazard loss.

        Args:
            hazard_logits: Predictions from hazard head (batch_size, 1)
            batch_labels: Dict containing 'cls' with class labels
            device: Device to run on
        """
        if hazard_logits is None or 'cls' not in batch_labels:
            return torch.tensor(0.0, device=device)

        # Get batch size
        batch_size = hazard_logits.shape[0]

        # Create hazard labels for each image in batch
        # 1 if image contains medical class, 0 otherwise
        hazard_labels = torch.zeros(batch_size, device=device)

        # Check which images contain medical objects
        cls_labels = batch_labels['cls']  # All class labels in batch
        batch_idx = batch_labels['batch_idx']  # Batch index for each label

        for i in range(batch_size):
            # Check if this image has any medical class objects
            image_mask = batch_idx == i
            if image_mask.any():
                image_classes = cls_labels[image_mask]
                if (image_classes == self.medical_idx).any():
                    hazard_labels[i] = 1.0

        # Calculate loss
        loss = self.bce_loss(hazard_logits.squeeze(-1), hazard_labels)

        return loss