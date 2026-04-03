# File: scripts/05_train.py
"""
Training script for local CPU/GPU training.

NOTE: This project was trained on Google Colab (see notebooks/02_training.ipynb).
This script is provided for reference and can be used for retraining locally
if you have a GPU.

For CPU training, expect ~6-8 hours with reduced settings.
For GPU training, expect ~40 minutes per model.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config
from src.unet import UNet
from src.attention_unet import AttentionUNet
from src.dataset import get_dataloaders
from src.losses import DiceBCELoss
from src.metrics import dice_coefficient, iou_score

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_NAME = Config.MODEL_NAME       # "UNet" or "AttentionUNet"
EPOCHS = Config.EPOCHS               # 50
LR = Config.LEARNING_RATE            # 0.0001
BATCH_SIZE = Config.BATCH_SIZE       # 8 for CPU, 16 for GPU
PATIENCE = Config.PATIENCE           # 10
DEVICE = Config.DEVICE

SAVE_DIR = os.path.join(Config.CHECKPOINT_DIR, MODEL_NAME.lower())
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"{'='*60}")
print(f"TRAINING: {MODEL_NAME}")
print(f"Device: {DEVICE}")
print(f"Epochs: {EPOCHS}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Learning Rate: {LR}")
print(f"{'='*60}")

# ============================================================
# DATA
# ============================================================
train_loader, val_loader, test_loader = get_dataloaders(
    data_path=Config.PROCESSED_DATA_PATH,
    batch_size=BATCH_SIZE,
    num_workers=Config.NUM_WORKERS
)

# ============================================================
# MODEL
# ============================================================
if MODEL_NAME == "UNet":
    model = UNet(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        features=Config.FEATURES
    ).to(DEVICE)
else:
    model = AttentionUNet(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        features=Config.FEATURES
    ).to(DEVICE)

params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params:,}")

# ============================================================
# LOSS, OPTIMIZER, SCHEDULER
# ============================================================
criterion = DiceBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=5, factor=0.5, verbose=True
)

# ============================================================
# TRAINING LOOP
# ============================================================
history = {
    'train_loss': [], 'train_dice': [],
    'val_loss': [], 'val_dice': [], 'val_iou': []
}

best_dice = 0
patience_counter = 0
start_time = time.time()

for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    train_loss_sum = 0
    train_dice_sum = 0
    n = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, masks in pbar:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss_sum += loss.item()
        train_dice_sum += dice_coefficient(outputs, masks)
        n += 1
        
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    # --- Validate ---
    model.eval()
    val_loss_sum = 0
    val_dice_sum = 0
    val_iou_sum = 0
    m = 0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            val_loss_sum += loss.item()
            val_dice_sum += dice_coefficient(outputs, masks)
            val_iou_sum += iou_score(outputs, masks)
            m += 1
    
    # --- Epoch Results ---
    tl = train_loss_sum / n
    td = train_dice_sum / n
    vl = val_loss_sum / m
    vd = val_dice_sum / m
    vi = val_iou_sum / m
    
    history['train_loss'].append(tl)
    history['train_dice'].append(td)
    history['val_loss'].append(vl)
    history['val_dice'].append(vd)
    history['val_iou'].append(vi)
    
    scheduler.step(vd)
    lr = optimizer.param_groups[0]['lr']
    
    print(f"  Train Loss: {tl:.4f} Dice: {td:.4f} | "
          f"Val Loss: {vl:.4f} Dice: {vd:.4f} IoU: {vi:.4f} | LR: {lr:.6f}")
    
    # Save best model
    if vd > best_dice:
        best_dice = vd
        patience_counter = 0
        torch.save({
            'model_name': MODEL_NAME,
            'model_state_dict': model.state_dict(),
            'features': Config.FEATURES,
            'in_channels': Config.IN_CHANNELS,
            'out_channels': Config.OUT_CHANNELS,
            'best_val_dice': best_dice,
            'epoch': epoch,
        }, os.path.join(SAVE_DIR, "best_model.pth"))
        print(f"  ✅ New best! Dice: {best_dice:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⛔ Early stopping at epoch {epoch+1}")
            break

total_time = time.time() - start_time
print(f"\n{'='*60}")
print(f"Training complete! Time: {total_time/60:.1f} min")
print(f"Best Dice: {best_dice:.4f}")
print(f"{'='*60}")

# Save history
np.save(os.path.join(SAVE_DIR, "history.npy"), history)