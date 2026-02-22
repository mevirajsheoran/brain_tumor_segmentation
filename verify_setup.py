# File: verify_setup.py
# Location: brain_tumor_segmentation/verify_setup.py

import sys
print(f"Python: {sys.version}")
print(f"Python path: {sys.executable}")
print()

libraries = {
    'torch': None,
    'torchvision': None,
    'nibabel': None,
    'numpy': None,
    'matplotlib': None,
    'sklearn': 'scikit-learn',
    'skimage': 'scikit-image',
    'albumentations': None,
    'tqdm': None,
    'pandas': None,
    'cv2': 'opencv-python',
    'streamlit': None,
    'PIL': 'Pillow',
}

print("Library Check:")
print("-" * 40)

all_ok = True
for import_name, pip_name in libraries.items():
    try:
        mod = __import__(import_name)
        version = getattr(mod, '__version__', 'OK')
        print(f"  ✅ {import_name:20s} {version}")
    except ImportError:
        install_name = pip_name if pip_name else import_name
        print(f"  ❌ {import_name:20s} MISSING — run: pip install {install_name}")
        all_ok = False

print()

# Check PyTorch details
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU: None (will use CPU or Google Colab)")

print()

# Check project structure
import os
required_folders = [
    'config', 'data/raw', 'data/processed', 'src',
    'scripts', 'checkpoints', 'results', 'app'
]

print("Folder Structure Check:")
print("-" * 40)
for folder in required_folders:
    exists = os.path.exists(folder)
    status = "✅" if exists else "❌"
    print(f"  {status} {folder}/")

print()
if all_ok:
    print("🎉 ALL LIBRARIES INSTALLED CORRECTLY!")
else:
    print("⚠️  Some libraries are missing. Install them before continuing.")