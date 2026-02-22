# File: scripts/03_preprocess.py

import os
import sys
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from tqdm import tqdm
import pickle
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

print("=" * 60)
print("STEP 3: DATASET PREPROCESSING")
print("=" * 60)

start_time = time.time()

data_path = Config.RAW_DATA_PATH
output_path = Config.PROCESSED_DATA_PATH
os.makedirs(output_path, exist_ok=True)

IMG_SIZE = Config.IMG_SIZE
MODALITY = Config.MODALITY
MIN_TUMOR = Config.MIN_TUMOR_PIXELS

# ============================================================
# 1. Get all patients
# ============================================================
all_patients = sorted([
    d for d in os.listdir(data_path)
    if os.path.isdir(os.path.join(data_path, d)) and d.startswith("BraTS")
])

print(f"Total patients available: {len(all_patients)}")

# Detect file extension
sample_path = os.path.join(data_path, all_patients[0])
sample_files = os.listdir(sample_path)
EXT = ".nii.gz" if any(f.endswith(".nii.gz") for f in sample_files) else ".nii"
print(f"File extension: {EXT}")

# Use subset for CPU
if Config.USE_SUBSET:
    np.random.seed(Config.RANDOM_SEED)
    selected = np.random.choice(
        all_patients,
        min(Config.MAX_PATIENTS, len(all_patients)),
        replace=False
    ).tolist()
    print(f"Using subset: {len(selected)} patients (CPU mode)")
else:
    selected = all_patients
    print(f"Using all {len(selected)} patients")

# ============================================================
# 2. Patient-wise split
# ============================================================
train_patients, temp = train_test_split(
    selected, test_size=0.3, random_state=Config.RANDOM_SEED
)
val_patients, test_patients = train_test_split(
    temp, test_size=0.5, random_state=Config.RANDOM_SEED
)

print(f"\nSplit:")
print(f"  Train: {len(train_patients)} patients")
print(f"  Val:   {len(val_patients)} patients")
print(f"  Test:  {len(test_patients)} patients")

# ============================================================
# 3. Helper functions
# ============================================================
def normalize(volume):
    """Normalize MRI volume to [0, 1]"""
    v = volume.astype(np.float32)
    mask = v > 0
    if mask.sum() == 0:
        return v
    v[mask] = (v[mask] - v[mask].min()) / (v[mask].max() - v[mask].min() + 1e-8)
    return v

def load_nifti(patient_path, patient_id, modality):
    """Load NIfTI file, trying both extensions"""
    for extension in [EXT, ".nii", ".nii.gz"]:
        path = os.path.join(patient_path, f"{patient_id}_{modality}{extension}")
        if os.path.exists(path):
            return nib.load(path).get_fdata()
    raise FileNotFoundError(f"Could not find {modality} for {patient_id}")

# ============================================================
# 4. Process patients
# ============================================================
def process_split(patient_list, split_name):
    images = []
    masks = []
    info = []
    errors = []
    
    print(f"\nProcessing {split_name} ({len(patient_list)} patients)...")
    
    for patient_id in tqdm(patient_list, desc=split_name):
        patient_path = os.path.join(data_path, patient_id)
        
        try:
            flair_vol = load_nifti(patient_path, patient_id, MODALITY)
            seg_vol = load_nifti(patient_path, patient_id, "seg")
        except Exception as e:
            errors.append(f"{patient_id}: {e}")
            continue
        
        # Normalize
        flair_vol = normalize(flair_vol)
        
        # Extract slices
        num_slices = flair_vol.shape[2]
        
        for s in range(num_slices):
            flair_slice = flair_vol[:, :, s]
            seg_slice = seg_vol[:, :, s]
            
            # Binary mask (tumor vs background)
            binary_mask = (seg_slice > 0).astype(np.float32)
            tumor_pixels = binary_mask.sum()
            
            # Keep tumor slices + small percentage of empty slices
            keep = False
            if tumor_pixels >= MIN_TUMOR:
                keep = True
            elif np.random.random() < Config.EMPTY_SLICE_RATIO:
                keep = True
            
            if keep:
                # Resize
                img = resize(flair_slice, (IMG_SIZE, IMG_SIZE),
                           preserve_range=True, anti_aliasing=True).astype(np.float32)
                msk = resize(binary_mask, (IMG_SIZE, IMG_SIZE),
                           preserve_range=True, order=0, anti_aliasing=False).astype(np.float32)
                
                # Ensure binary
                msk = (msk > 0.5).astype(np.float32)
                
                images.append(img)
                masks.append(msk)
                info.append({
                    'patient': patient_id,
                    'slice': s,
                    'tumor_pixels': int(tumor_pixels)
                })
    
    if errors:
        print(f"  Errors ({len(errors)}):")
        for e in errors[:5]:
            print(f"    {e}")
    
    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)
    
    tumor_count = sum(1 for i in info if i['tumor_pixels'] > 0)
    empty_count = len(info) - tumor_count
    
    print(f"  Result: {len(images)} slices")
    print(f"    Shape: {images.shape}")
    print(f"    Tumor slices: {tumor_count}")
    print(f"    Empty slices: {empty_count}")
    print(f"    Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"    Mask unique: {np.unique(masks)}")
    
    return images, masks, info

# Process each split
train_images, train_masks, train_info = process_split(train_patients, "train")
val_images, val_masks, val_info = process_split(val_patients, "val")
test_images, test_masks, test_info = process_split(test_patients, "test")

# ============================================================
# 5. Save everything
# ============================================================
print(f"\nSaving processed data...")

np.save(os.path.join(output_path, "train_images.npy"), train_images)
np.save(os.path.join(output_path, "train_masks.npy"), train_masks)
np.save(os.path.join(output_path, "val_images.npy"), val_images)
np.save(os.path.join(output_path, "val_masks.npy"), val_masks)
np.save(os.path.join(output_path, "test_images.npy"), test_images)
np.save(os.path.join(output_path, "test_masks.npy"), test_masks)

metadata = {
    'train_patients': train_patients,
    'val_patients': val_patients,
    'test_patients': test_patients,
    'train_info': train_info,
    'val_info': val_info,
    'test_info': test_info,
    'img_size': IMG_SIZE,
    'modality': MODALITY,
    'file_extension': EXT
}

with open(os.path.join(output_path, "metadata.pkl"), "wb") as f:
    pickle.dump(metadata, f)

# Print file sizes
total_mb = 0
for fname in os.listdir(output_path):
    fpath = os.path.join(output_path, fname)
    size_mb = os.path.getsize(fpath) / (1024 * 1024)
    total_mb += size_mb
    print(f"  {fname}: {size_mb:.1f} MB")

elapsed = time.time() - start_time

print(f"\n{'='*60}")
print(f"✅ PREPROCESSING COMPLETE!")
print(f"{'='*60}")
print(f"Time taken: {elapsed/60:.1f} minutes")
print(f"Total disk usage: {total_mb:.1f} MB")
print(f"Files saved to: {os.path.abspath(output_path)}")
print(f"\nSummary:")
print(f"  Train: {train_images.shape[0]} slices ({train_images.shape})")
print(f"  Val:   {val_images.shape[0]} slices ({val_images.shape})")
print(f"  Test:  {test_images.shape[0]} slices ({test_images.shape})")
print(f"\nNext step: python scripts/04_verify_preprocessing.py")