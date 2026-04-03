# File: scripts/07_compare_models.py
"""
Compare U-Net and Attention U-Net side by side.
Produces comparison table and visualization.
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config
from src.unet import UNet
from src.attention_unet import AttentionUNet
from src.metrics import compute_all_metrics

DEVICE = Config.DEVICE
SAVE_DIR = os.path.join(Config.RESULTS_DIR, "comparison")
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# LOAD BOTH MODELS
# ============================================================
def load_model(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    features = ckpt.get('features', [64, 128, 256, 512])
    name = ckpt.get('model_name', 'UNet')
    
    if 'Attention' in name:
        model = AttentionUNet(in_channels=1, out_channels=1, features=features)
    else:
        model = UNet(in_channels=1, out_channels=1, features=features)
    
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, name

# Load test data
test_images = np.load(os.path.join(Config.PROCESSED_DATA_PATH, "test_images.npy"))
test_masks = np.load(os.path.join(Config.PROCESSED_DATA_PATH, "test_masks.npy"))

# Find and load models
ckpt_dir = Config.CHECKPOINT_DIR
models = {}

for f in os.listdir(ckpt_dir):
    if f.endswith('.pth'):
        try:
            m, n = load_model(os.path.join(ckpt_dir, f))
            models[n] = {'model': m, 'params': sum(p.numel() for p in m.parameters())}
            print(f"✅ Loaded {n} from {f}")
        except Exception as e:
            print(f"⚠️  Skipped {f}: {e}")

if len(models) < 2:
    print("Need at least 2 models for comparison!")
    sys.exit(1)

# ============================================================
# EVALUATE BOTH
# ============================================================
results = {}

for name, info in models.items():
    model = info['model']
    all_dice = []
    
    with torch.no_grad():
        for i in range(0, len(test_images), 16):
            batch_imgs = test_images[i:i+16]
            batch_masks = test_masks[i:i+16]
            
            imgs_t = torch.FloatTensor(batch_imgs).unsqueeze(1)
            masks_t = torch.FloatTensor(batch_masks).unsqueeze(1)
            
            outputs = model(imgs_t)
            
            for j in range(imgs_t.shape[0]):
                metrics = compute_all_metrics(outputs[j:j+1], masks_t[j:j+1])
                all_dice.append(metrics)
    
    results[name] = {
        'dice': np.mean([m['dice'] for m in all_dice]),
        'iou': np.mean([m['iou'] for m in all_dice]),
        'precision': np.mean([m['precision'] for m in all_dice]),
        'recall': np.mean([m['recall'] for m in all_dice]),
        'params': info['params'],
    }

# ============================================================
# COMPARISON TABLE
# ============================================================
print(f"\n{'='*60}")
print(f"MODEL COMPARISON")
print(f"{'='*60}")

df = pd.DataFrame(results).T
df.to_csv(os.path.join(SAVE_DIR, "comparison_table.csv"))
print(df.to_string())
print(f"\n✅ Saved: {SAVE_DIR}/comparison_table.csv")