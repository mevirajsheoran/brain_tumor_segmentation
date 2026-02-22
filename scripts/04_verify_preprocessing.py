# File: scripts/04_verify_preprocessing.py

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

print("=" * 60)
print("STEP 4: VERIFY PREPROCESSING")
print("=" * 60)

processed = Config.PROCESSED_DATA_PATH
results = os.path.join(Config.RESULTS_DIR, "visualizations")
os.makedirs(results, exist_ok=True)

# Load data
train_images = np.load(os.path.join(processed, "train_images.npy"))
train_masks = np.load(os.path.join(processed, "train_masks.npy"))
val_images = np.load(os.path.join(processed, "val_images.npy"))
val_masks = np.load(os.path.join(processed, "val_masks.npy"))
test_images = np.load(os.path.join(processed, "test_images.npy"))
test_masks = np.load(os.path.join(processed, "test_masks.npy"))

print(f"\nDataset Shapes:")
print(f"  Train: images={train_images.shape}, masks={train_masks.shape}")
print(f"  Val:   images={val_images.shape},   masks={val_masks.shape}")
print(f"  Test:  images={test_images.shape},  masks={test_masks.shape}")

print(f"\nData Types:")
print(f"  Images: {train_images.dtype}")
print(f"  Masks:  {train_masks.dtype}")

print(f"\nValue Ranges:")
print(f"  Images: [{train_images.min():.4f}, {train_images.max():.4f}]")
print(f"  Masks:  unique values = {np.unique(train_masks)}")

# Count tumor vs empty
train_tumor = np.sum(train_masks.reshape(len(train_masks), -1).sum(axis=1) > 0)
train_empty = len(train_masks) - train_tumor
print(f"\nTrain Distribution:")
print(f"  Tumor slices: {train_tumor} ({train_tumor/len(train_masks)*100:.1f}%)")
print(f"  Empty slices: {train_empty} ({train_empty/len(train_masks)*100:.1f}%)")

# ============================================================
# Visualize random samples
# ============================================================
np.random.seed(42)

# Get indices with tumor
tumor_indices = np.where(
    train_masks.reshape(len(train_masks), -1).sum(axis=1) > 100
)[0]

if len(tumor_indices) < 6:
    tumor_indices = np.where(
        train_masks.reshape(len(train_masks), -1).sum(axis=1) > 0
    )[0]

selected = np.random.choice(tumor_indices, min(6, len(tumor_indices)), replace=False)

fig, axes = plt.subplots(len(selected), 3, figsize=(12, 4 * len(selected)))

if len(selected) == 1:
    axes = axes.reshape(1, -1)

for row, idx in enumerate(selected):
    img = train_images[idx]
    msk = train_masks[idx]
    
    axes[row, 0].imshow(img, cmap='gray')
    axes[row, 0].set_title(f'MRI (idx={idx})', fontsize=12)
    axes[row, 0].axis('off')
    
    axes[row, 1].imshow(msk, cmap='gray')
    axes[row, 1].set_title(f'Mask (pixels={int(msk.sum())})', fontsize=12)
    axes[row, 1].axis('off')
    
    axes[row, 2].imshow(img, cmap='gray')
    axes[row, 2].imshow(msk, cmap='Reds', alpha=0.4)
    axes[row, 2].set_title('Overlay', fontsize=12)
    axes[row, 2].axis('off')

plt.suptitle('Preprocessed Data Verification', fontsize=16)
plt.tight_layout()

save_path = os.path.join(results, "preprocessing_verify.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✅ Verification image saved: {save_path}")
print(f"Open this image to visually confirm data looks correct!")

# ============================================================
# Final checks
# ============================================================
all_pass = True

if train_images.shape[1] != Config.IMG_SIZE or train_images.shape[2] != Config.IMG_SIZE:
    print(f"❌ Image size mismatch! Expected {Config.IMG_SIZE}, got {train_images.shape[1:]}")
    all_pass = False

if not np.array_equal(np.unique(train_masks), np.array([0., 1.])):
    if not np.array_equal(np.unique(train_masks), np.array([0.])):
        possible = np.unique(train_masks)
        if len(possible) > 2:
            print(f"❌ Mask not binary! Values: {possible}")
            all_pass = False

if train_images.max() > 1.01 or train_images.min() < -0.01:
    print(f"⚠️  Image values outside [0,1]: [{train_images.min():.3f}, {train_images.max():.3f}]")

if all_pass:
    print(f"\n✅ ALL CHECKS PASSED!")
    print(f"\n{'='*60}")
    print(f"PREPROCESSING IS DONE. NEXT STEPS:")
    print(f"{'='*60}")
    print(f"\nOption A: Train locally (CPU, slow)")
    print(f"  python scripts/05_train.py")
    print(f"\nOption B: Train on Google Colab (RECOMMENDED)")
    print(f"  1. Upload data/processed/ folder to Google Drive")
    print(f"  2. Open Colab notebook")
    print(f"  3. Train with GPU")