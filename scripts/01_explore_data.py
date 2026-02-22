# File: scripts/01_explore_data.py

import os
import sys
import nibabel as nib
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

print("=" * 60)
print("STEP 1: DATASET EXPLORATION")
print("=" * 60)

data_path = Config.RAW_DATA_PATH

# ============================================================
# 1. List all patients
# ============================================================
all_patients = sorted([
    d for d in os.listdir(data_path)
    if os.path.isdir(os.path.join(data_path, d)) and d.startswith("BraTS")
])

print(f"\nTotal patients: {len(all_patients)}")
print(f"First 5: {all_patients[:5]}")
print(f"Last 5:  {all_patients[-5:]}")

# ============================================================
# 2. Examine one patient's files
# ============================================================
sample_patient = all_patients[1]  # Skip first, sometimes it's different
patient_path = os.path.join(data_path, sample_patient)

print(f"\n--- Files for {sample_patient} ---")
for f in sorted(os.listdir(patient_path)):
    filepath = os.path.join(patient_path, f)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  {f:50s} {size_mb:.2f} MB")

# ============================================================
# 3. Load and examine one volume
# ============================================================
print(f"\n--- Loading MRI volumes for {sample_patient} ---")

# Detect file extension (.nii or .nii.gz)
files = os.listdir(patient_path)
ext = ".nii.gz" if any(f.endswith(".nii.gz") for f in files) else ".nii"
print(f"File extension: {ext}")

modalities = {}
for mod_name in ["flair", "t1", "t1ce", "t2", "seg"]:
    filepath = os.path.join(patient_path, f"{sample_patient}_{mod_name}{ext}")
    
    if not os.path.exists(filepath):
        # Try alternate extensions
        alt_ext = ".nii" if ext == ".nii.gz" else ".nii.gz"
        filepath = os.path.join(patient_path, f"{sample_patient}_{mod_name}{alt_ext}")
    
    if os.path.exists(filepath):
        vol = nib.load(filepath).get_fdata()
        modalities[mod_name] = vol
        print(f"  {mod_name:6s}: shape={vol.shape}, dtype={vol.dtype}, "
              f"min={vol.min():.2f}, max={vol.max():.2f}")
    else:
        print(f"  {mod_name:6s}: ❌ FILE NOT FOUND")

# ============================================================
# 4. Analyze segmentation mask
# ============================================================
if "seg" in modalities:
    seg = modalities["seg"]
    unique_labels = np.unique(seg)
    print(f"\n--- Segmentation Mask Analysis ---")
    print(f"Unique labels: {unique_labels}")
    print(f"Label meanings:")
    print(f"  0 = Background (no tumor)")
    print(f"  1 = Necrotic / Non-enhancing tumor core")
    print(f"  2 = Peritumoral edema")
    print(f"  4 = GD-enhancing tumor")
    
    for label in unique_labels:
        count = np.sum(seg == label)
        percentage = (count / seg.size) * 100
        print(f"  Label {int(label)}: {count:,} voxels ({percentage:.2f}%)")

# ============================================================
# 5. Count tumor slices
# ============================================================
if "seg" in modalities:
    seg = modalities["seg"]
    total_slices = seg.shape[2]
    tumor_slices = 0
    
    for s in range(total_slices):
        if np.any(seg[:, :, s] > 0):
            tumor_slices += 1
    
    print(f"\n--- Slice Analysis ---")
    print(f"Total slices per patient: {total_slices}")
    print(f"Slices WITH tumor: {tumor_slices}")
    print(f"Slices WITHOUT tumor: {total_slices - tumor_slices}")
    print(f"Tumor slice ratio: {tumor_slices/total_slices:.2%}")

# ============================================================
# 6. Check consistency across patients
# ============================================================
print(f"\n--- Checking first 10 patients for consistency ---")
for patient in all_patients[:10]:
    ppath = os.path.join(data_path, patient)
    flair_file = os.path.join(ppath, f"{patient}_flair{ext}")
    
    if not os.path.exists(flair_file):
        alt_ext = ".nii" if ext == ".nii.gz" else ".nii.gz"
        flair_file = os.path.join(ppath, f"{patient}_flair{alt_ext}")
    
    if os.path.exists(flair_file):
        vol = nib.load(flair_file).get_fdata()
        print(f"  {patient}: shape={vol.shape}")
    else:
        print(f"  {patient}: ❌ flair file not found")

print(f"\n✅ Exploration complete!")
print(f"Proceed to: python scripts/02_visualize_modalities.py")