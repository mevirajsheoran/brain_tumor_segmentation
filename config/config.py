# File: config/config.py

import os
import torch


class Config:
    # ============================================================
    # PATHS — UPDATE THESE IF YOUR FOLDER STRUCTURE IS DIFFERENT
    # ============================================================
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # After extraction, check what path extract_dataset.py printed
    # and update this if needed
    RAW_DATA_PATH = os.path.join(
        PROJECT_ROOT, "data", "raw",
        "BraTS2020_TrainingData",
        "MICCAI_BraTS2020_TrainingData"
    )
    
    PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    
    # ============================================================
    # DATA PARAMETERS
    # ============================================================
    IMG_SIZE = 128
    MODALITY = "flair"
    MIN_TUMOR_PIXELS = 50
    EMPTY_SLICE_RATIO = 0.1
    
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    RANDOM_SEED = 42
    
    # ============================================================
    # MODEL PARAMETERS
    # ============================================================
    MODEL_NAME = "UNet"
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    FEATURES = [32, 64, 128, 256]     # Smaller for CPU
    
    # ============================================================
    # TRAINING PARAMETERS
    # ============================================================
    BATCH_SIZE = 8
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    PATIENCE = 10
    NUM_WORKERS = 0                    # MUST be 0 for CPU on Windows
    
    # ============================================================
    # CPU OPTIMIZATION
    # ============================================================
    MAX_PATIENTS = 80                  # Use subset for CPU training
    USE_SUBSET = True                  # Set False if using GPU
    
    # ============================================================
    # DEVICE
    # ============================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @classmethod
    def verify(cls):
        """Verify all paths and settings"""
        print("=" * 60)
        print("CONFIGURATION VERIFICATION")
        print("=" * 60)
        
        # Check paths
        print(f"\nProject Root: {cls.PROJECT_ROOT}")
        print(f"  Exists: {os.path.exists(cls.PROJECT_ROOT)}")
        
        print(f"\nRaw Data: {cls.RAW_DATA_PATH}")
        exists = os.path.exists(cls.RAW_DATA_PATH)
        print(f"  Exists: {exists}")
        
        if exists:
            patients = [d for d in os.listdir(cls.RAW_DATA_PATH) 
                       if d.startswith("BraTS")]
            print(f"  Patients found: {len(patients)}")
        else:
            print("  ❌ DATA PATH NOT FOUND!")
            print("  Check if extraction completed correctly")
            print("  Look inside data/raw/ for the correct folder name")
            
            # Try to find it
            raw_path = os.path.join(cls.PROJECT_ROOT, "data", "raw")
            if os.path.exists(raw_path):
                print(f"\n  Contents of data/raw/:")
                for item in os.listdir(raw_path):
                    print(f"    {item}/")
                    subpath = os.path.join(raw_path, item)
                    if os.path.isdir(subpath):
                        for sub in os.listdir(subpath)[:3]:
                            print(f"      {sub}/")
        
        print(f"\nProcessed Data: {cls.PROCESSED_DATA_PATH}")
        print(f"  Exists: {os.path.exists(cls.PROCESSED_DATA_PATH)}")
        
        print(f"\nDevice: {cls.DEVICE}")
        print(f"Model: {cls.MODEL_NAME}")
        print(f"Features: {cls.FEATURES}")
        print(f"Image Size: {cls.IMG_SIZE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Max Patients: {cls.MAX_PATIENTS}")
        
        print("=" * 60)
        
        return exists


if __name__ == "__main__":
    data_found = Config.verify()
    
    if not data_found:
        print("\n⚠️  FIX THE DATA PATH BEFORE CONTINUING")
        print("Edit config/config.py → RAW_DATA_PATH")
    else:
        print("\n✅ Config is correct! Proceed to next step.")