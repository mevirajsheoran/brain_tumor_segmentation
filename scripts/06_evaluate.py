# File: scripts/06_evaluate.py
"""
Evaluate trained model on test set.
Produces metrics and prediction visualizations.
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config
from src.unet import UNet
from src.attention_unet import AttentionUNet
from src.dataset import get_dataloaders
from src.metrics import compute_all_metrics

DEVICE = Config.DEVICE
RESULTS_DIR = os.path.join(Config.RESULTS_DIR, "predictions")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# LOAD MODEL
# ============================================================
def load_model(checkpoint_path):
    """Load model from checkpoint"""
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    model_name = ckpt.get('model_name', 'UNet')
    features = ckpt.get('features', [64, 128, 256, 512])
    
    if 'Attention' in model_name:
        model = AttentionUNet(in_channels=1, out_channels=1, features=features)
    else:
        model = UNet(in_channels=1, out_channels=1, features=features)
    
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model.to(DEVICE)
    
    return model, model_name

# ============================================================
# EVALUATE
# ============================================================
def evaluate(model, test_loader, model_name):
    """Run evaluation on test set"""
    all_metrics = []
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            outputs = model(images)
            
            for i in range(images.shape[0]):
                m = compute_all_metrics(outputs[i:i+1], masks[i:i+1])
                all_metrics.append(m)
    
    # Average
    results = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        results[f"{key}_mean"] = np.mean(values)
        results[f"{key}_std"] = np.std(values)
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS: {model_name}")
    print(f"{'='*60}")
    print(f"  Dice:      {results['dice_mean']:.4f} ± {results['dice_std']:.4f}")
    print(f"  IoU:       {results['iou_mean']:.4f} ± {results['iou_std']:.4f}")
    print(f"  Precision: {results['precision_mean']:.4f}")
    print(f"  Recall:    {results['recall_mean']:.4f}")
    print(f"  Accuracy:  {results['accuracy_mean']:.4f}")
    print(f"  Samples:   {len(all_metrics)}")
    print(f"{'='*60}")
    
    return results

# ============================================================
# MAIN
# ============================================================
_, _, test_loader = get_dataloaders(Config.PROCESSED_DATA_PATH, batch_size=16)

# Evaluate all models found in checkpoints
checkpoint_dir = Config.CHECKPOINT_DIR
pth_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]

for pth_file in pth_files:
    path = os.path.join(checkpoint_dir, pth_file)
    try:
        model, name = load_model(path)
        results = evaluate(model, test_loader, name)
    except Exception as e:
        print(f"Error evaluating {pth_file}: {e}")

print("\n✅ Evaluation complete!")