# File: src/losses.py

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    
    Why Dice Loss?
    - Standard Cross Entropy fails with class imbalance
    - Only 0.75% of pixels are tumor → model predicts all background
    - Dice Loss directly measures overlap, penalizing lazy predictions
    
    Formula: DiceLoss = 1 - (2 * |Pred ∩ GT|) / (|Pred| + |GT|)
    - Perfect prediction → Dice = 1 → Loss = 0
    - No overlap → Dice = 0 → Loss = 1
    """
    
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth    # Prevents division by zero
    
    def forward(self, predictions, targets):
        # Apply sigmoid to convert logits → probabilities (0 to 1)
        predictions = torch.sigmoid(predictions)
        
        # Flatten to 1D vectors
        pred_flat = predictions.view(-1)
        target_flat = targets.view(-1)
        
        # Calculate Dice coefficient
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        
        return 1 - dice     # Loss = 1 - Dice


class DiceBCELoss(nn.Module):
    """
    Combined Dice Loss + Binary Cross Entropy Loss.
    
    Why combine?
    - Dice Loss: Handles class imbalance, optimizes overlap
    - BCE Loss: Provides stable gradients, pixel-level accuracy
    - Together: Best of both worlds
    
    We use 50% Dice + 50% BCE by default.
    """
    
    def __init__(self, smooth=1e-6, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.smooth = smooth
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()   # Numerically stable BCE
    
    def forward(self, predictions, targets):
        # BCE Loss (works on raw logits, applies sigmoid internally)
        bce_loss = self.bce(predictions, targets)
        
        # Dice Loss (need explicit sigmoid)
        pred_sigmoid = torch.sigmoid(predictions)
        pred_flat = pred_sigmoid.view(-1)
        target_flat = targets.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        dice_loss = 1 - dice
        
        # Weighted combination
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss