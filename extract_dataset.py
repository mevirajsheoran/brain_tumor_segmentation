# File: extract_dataset.py
# Location: brain_tumor_segmentation/extract_dataset.py

import zipfile
import os
import shutil


# ============================================================
# CHANGE THIS PATH to where your downloaded zip file is
# ============================================================
ZIP_FILE_PATH = r"C:\Users\viraj\Downloads\brats20-dataset-training-validation.zip"
# Mac/Linux example:
# ZIP_FILE_PATH = "/Users/yourname/Downloads/brats20-dataset-training-validation.zip"

EXTRACT_TO = os.path.join("data", "raw")

# ============================================================
# Check zip file exists
# ============================================================
if not os.path.exists(ZIP_FILE_PATH):
    print(f"❌ ZIP file not found at: {ZIP_FILE_PATH}")
    print(f"\nPlease check:")
    print(f"  1. The file name is correct")
    print(f"  2. The path is correct")
    print(f"  3. The file has finished downloading")
    print(f"\nYour Downloads folder contains:")
    
    downloads = os.path.expanduser("~/Downloads")
    if os.path.exists(downloads):
        for f in os.listdir(downloads):
            if 'brats' in f.lower() or 'brain' in f.lower():
                print(f"  Found: {os.path.join(downloads, f)}")
    exit(1)

# ============================================================
# Extract
# ============================================================
zip_size = os.path.getsize(ZIP_FILE_PATH) / (1024 * 1024 * 1024)
print(f"ZIP file found: {ZIP_FILE_PATH}")
print(f"Size: {zip_size:.2f} GB")
print(f"Extracting to: {os.path.abspath(EXTRACT_TO)}")
print(f"This will take 5-15 minutes depending on your disk speed...")
print()

os.makedirs(EXTRACT_TO, exist_ok=True)

with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
    total_files = len(zip_ref.namelist())
    print(f"Total files in zip: {total_files}")
    
    for i, file in enumerate(zip_ref.namelist()):
        zip_ref.extract(file, EXTRACT_TO)
        
        if (i + 1) % 500 == 0 or (i + 1) == total_files:
            progress = ((i + 1) / total_files) * 100
            print(f"  Extracting: {progress:.0f}% ({i+1}/{total_files})")

print(f"\n✅ Extraction complete!")

# ============================================================
# Verify extraction
# ============================================================
# Find the actual data folder (might be nested)
def find_patient_folder(base_path):
    """Find the folder containing BraTS patient directories"""
    for root, dirs, files in os.walk(base_path):
        brats_dirs = [d for d in dirs if d.startswith("BraTS20_Training")]
        if len(brats_dirs) > 10:
            return root
    return None

data_folder = find_patient_folder(EXTRACT_TO)

if data_folder:
    patients = [d for d in os.listdir(data_folder) if d.startswith("BraTS")]
    print(f"\nData folder found: {data_folder}")
    print(f"Number of patients: {len(patients)}")
    
    # Show first patient's files
    if patients:
        sample = os.path.join(data_folder, sorted(patients)[0])
        print(f"\nSample patient ({sorted(patients)[0]}):")
        for f in os.listdir(sample):
            fpath = os.path.join(sample, f)
            fsize = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  {f} ({fsize:.1f} MB)")
    
    print(f"\n📝 IMPORTANT: Update your config.py with this path:")
    print(f'   RAW_DATA_PATH = r"{data_folder}"')
else:
    print("⚠️  Could not auto-detect patient folders.")
    print(f"Check inside: {os.path.abspath(EXTRACT_TO)}")
    print("Look for folders named BraTS20_Training_001, etc.")