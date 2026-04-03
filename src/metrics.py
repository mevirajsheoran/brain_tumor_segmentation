# File: src/metrics.py

import torch
import numpy as np


def dice_coefficient(predictions, targets, threshold=0.5, smooth=1e-6):
    """
    Dice Coefficient — PRIMARY evaluation metric.
    
    Measures overlap between predicted mask and ground truth.
    Range: 0 (no overlap) to 1 (perfect overlap)
    
    Good values for brain tumor segmentation:
    - < 0.5: Poor
    - 0.5 - 0.7: Decent
    - 0.7 - 0.85: Good
    - > 0.85: Excellent
    """
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()
    
    pred_flat = predictions.view(-1)
    target_flat = targets.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (
        pred_flat.sum() + target_flat.sum() + smooth
    )
    
    return dice.item()


def iou_score(predictions, targets, threshold=0.5, smooth=1e-6):
    """
    IoU (Intersection over Union) — also called Jaccard Index.
    
    IoU = Intersection / Union
    Always lower than Dice for the same prediction.
    
    Relationship: Dice = 2*IoU / (1+IoU)
    """
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()
    
    pred_flat = predictions.view(-1)
    target_flat = targets.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def precision_score(predictions, targets, threshold=0.5, smooth=1e-6):
    """
    Precision = TP / (TP + FP)
    
    "Of all pixels predicted as tumor, how many are actually tumor?"
    High precision = few false alarms
    """
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()
    
    tp = (predictions * targets).sum()
    fp = (predictions * (1 - targets)).sum()
    
    precision = (tp + smooth) / (tp + fp + smooth)
    return precision.item()


def recall_score(predictions, targets, threshold=0.5, smooth=1e-6):
    """
    Recall = TP / (TP + FN)
    
    "Of all actual tumor pixels, how many did we detect?"
    High recall = few missed tumors
    
    In medical imaging, recall is often more important than precision
    because missing a tumor is worse than a false alarm.
    """
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()
    
    tp = (predictions * targets).sum()
    fn = ((1 - predictions) * targets).sum()
    
    recall = (tp + smooth) / (tp + fn + smooth)
    return recall.item()


def pixel_accuracy(predictions, targets, threshold=0.5):
    """
    Pixel Accuracy = Correct Pixels / Total Pixels
    
    WARNING: This metric is misleading for segmentation!
    With 99% background, predicting "all background" gives 99% accuracy.
    Always use Dice/IoU as primary metrics.
    """
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()
    
    correct = (predictions == targets).float().sum()
    total = targets.numel()
    
    return (correct / total).item()


def compute_all_metrics(predictions, targets, threshold=0.5):
    """Compute all metrics at once. Returns a dictionary."""
    return {
        'dice': dice_coefficient(predictions, targets, threshold),
        'iou': iou_score(predictions, targets, threshold),
        'precision': precision_score(predictions, targets, threshold),
        'recall': recall_score(predictions, targets, threshold),
        'accuracy': pixel_accuracy(predictions, targets, threshold),
    }