# File: scripts/02_visualize_modalities.py

import os
import sys
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')          # Use non-interactive backend (works without display)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

print("=" * 60)
print("STEP 2: DATA VISUALIZATION")
print("=" * 60)

data_path = Config.RAW_DATA_PATH
results_path = os.path.join(Config.RESULTS_DIR, "visualizations")
os.makedirs(results_path, exist_ok=True)

# ============================================================
# Load one patient
# ============================================================
patients = sorted([d for d in os.listdir(data_path) if d.startswith("BraTS")])
patient_id = patients[1]
patient_path = os.path.join(data_path, patient_id)

# Detect extension
files = os.listdir(patient_path)
ext = ".nii.gz" if any(f.endswith(".nii.gz") for f in files) else ".nii"

print(f"Loading patient: {patient_id}")

def load_volume(patient_path, patient_id, modality, ext):
    """Load a NIfTI volume, trying both extensions"""
    path = os.path.join(patient_path, f"{patient_id}_{modality}{ext}")
    if not os.path.exists(path):
        alt = ".nii" if ext == ".nii.gz" else ".nii.gz"
        path = os.path.join(patient_path, f"{patient_id}_{modality}{alt}")
    return nib.load(path).get_fdata()

flair = load_volume(patient_path, patient_id, "flair", ext)
t1 = load_volume(patient_path, patient_id, "t1", ext)
t1ce = load_volume(patient_path, patient_id, "t1ce", ext)
t2 = load_volume(patient_path, patient_id, "t2", ext)
seg = load_volume(patient_path, patient_id, "seg", ext)

print(f"Volume shape: {flair.shape}")

# ============================================================
# Find best slice (most tumor)
# ============================================================
best_slice = 0
max_tumor = 0
for i in range(seg.shape[2]):
    tumor_count = np.sum(seg[:, :, i] > 0)
    if tumor_count > max_tumor:
        max_tumor = tumor_count
        best_slice = i

print(f"Best slice: {best_slice} ({max_tumor} tumor pixels)")

# ============================================================
# PLOT 1: All modalities side by side
# ============================================================
s = best_slice

fig, axes = plt.subplots(1, 5, figsize=(25, 5))

axes[0].imshow(flair[:, :, s], cmap='gray')
axes[0].set_title('FLAIR', fontsize=14)
axes[0].axis('off')

axes[1].imshow(t1[:, :, s], cmap='gray')
axes[1].set_title('T1', fontsize=14)
axes[1].axis('off')

axes[2].imshow(t1ce[:, :, s], cmap='gray')
axes[2].set_title('T1ce', fontsize=14)
axes[2].axis('off')

axes[3].imshow(t2[:, :, s], cmap='gray')
axes[3].set_title('T2', fontsize=14)
axes[3].axis('off')

axes[4].imshow(seg[:, :, s], cmap='nipy_spectral')
axes[4].set_title('Segmentation', fontsize=14)
axes[4].axis('off')

plt.suptitle(f'{patient_id} — Slice {s}', fontsize=16)
plt.tight_layout()

save_path = os.path.join(results_path, "modalities_view.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {save_path}")

# ============================================================
# PLOT 2: MRI with mask overlay
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(flair[:, :, s], cmap='gray')
axes[0].set_title('FLAIR MRI', fontsize=14)
axes[0].axis('off')

# Color the mask
mask = seg[:, :, s].copy()
mask_colored = np.zeros((*mask.shape, 3))
mask_colored[mask == 1] = [1, 0, 0]    # Red: Necrotic
mask_colored[mask == 2] = [0, 1, 0]    # Green: Edema
mask_colored[mask == 4] = [1, 1, 0]    # Yellow: Enhancing

axes[1].imshow(mask_colored)
axes[1].set_title('Tumor Mask\n(Red=Necrotic, Green=Edema, Yellow=Enhancing)', fontsize=12)
axes[1].axis('off')

axes[2].imshow(flair[:, :, s], cmap='gray')
axes[2].imshow(mask_colored, alpha=0.4)
axes[2].set_title('Overlay', fontsize=14)
axes[2].axis('off')

plt.suptitle(f'{patient_id} — Slice {s}', fontsize=16)
plt.tight_layout()

save_path = os.path.join(results_path, "overlay_view.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {save_path}")

# ============================================================
# PLOT 3: Multiple slices showing tumor progression
# ============================================================
# Find slices with tumor
tumor_slices = []
for i in range(seg.shape[2]):
    count = np.sum(seg[:, :, i] > 0)
    if count > 100:
        tumor_slices.append((i, count))

tumor_slices.sort(key=lambda x: x[1])

if len(tumor_slices) >= 8:
    # Pick 8 evenly spaced slices
    indices = np.linspace(0, len(tumor_slices) - 1, 8, dtype=int)
    selected = [tumor_slices[i][0] for i in indices]
else:
    selected = [t[0] for t in tumor_slices[:8]]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for idx, slice_num in enumerate(selected):
    row = idx // 4
    col = idx % 4
    
    axes[row, col].imshow(flair[:, :, slice_num], cmap='gray')
    
    mask = seg[:, :, slice_num]
    mask_vis = np.zeros((*mask.shape, 4))
    mask_vis[mask > 0] = [1, 0, 0, 0.4]
    axes[row, col].imshow(mask_vis)
    
    tumor_px = np.sum(mask > 0)
    axes[row, col].set_title(f'Slice {slice_num}\n({tumor_px} tumor px)', fontsize=11)
    axes[row, col].axis('off')

plt.suptitle(f'{patient_id} — Tumor Across Slices', fontsize=16)
plt.tight_layout()

save_path = os.path.join(results_path, "tumor_progression.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {save_path}")

print(f"\n✅ All visualizations saved to: {results_path}")
print(f"Open these images to see your data!")
print(f"Proceed to: python scripts/03_preprocess.py")