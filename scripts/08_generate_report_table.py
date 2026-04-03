# File: scripts/08_generate_report_table.py
"""
Generate literature survey table and project summary for report.
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

SAVE_DIR = os.path.join(Config.RESULTS_DIR, "comparison")
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# LITERATURE SURVEY TABLE
# ============================================================
papers = [
    {
        "Paper": "Ronneberger et al. (2015)",
        "Title": "U-Net: Convolutional Networks for Biomedical Image Segmentation",
        "Model": "U-Net",
        "Dataset": "ISBI Cell Tracking",
        "Dice Score": "0.920",
        "Loss Function": "Cross Entropy + Weight Map",
        "Key Contribution": "Skip connections for precise localization"
    },
    {
        "Paper": "Oktay et al. (2018)",
        "Title": "Attention U-Net: Learning Where to Look for the Pancreas",
        "Model": "Attention U-Net",
        "Dataset": "CT Abdomen",
        "Dice Score": "0.847",
        "Loss Function": "Dice Loss",
        "Key Contribution": "Attention gates to focus on relevant regions"
    },
    {
        "Paper": "Isensee et al. (2021)",
        "Title": "nnU-Net: A Self-configuring Method for Deep Learning-based Biomedical Image Segmentation",
        "Model": "nnU-Net",
        "Dataset": "BraTS 2020",
        "Dice Score": "0.885",
        "Loss Function": "Dice + CE",
        "Key Contribution": "Auto-configured pipeline, state-of-the-art"
    },
    {
        "Paper": "Myronenko (2019)",
        "Title": "3D MRI Brain Tumor Segmentation Using Autoencoder Regularization",
        "Model": "3D Encoder-Decoder + VAE",
        "Dataset": "BraTS 2018",
        "Dice Score": "0.884",
        "Loss Function": "Dice + L2 + KL",
        "Key Contribution": "BraTS 2018 winner, VAE regularization"
    },
    {
        "Paper": "This Project",
        "Title": "Brain Tumor Segmentation using U-Net and Attention U-Net",
        "Model": "U-Net + Attention U-Net",
        "Dataset": "BraTS 2020 (80 patients)",
        "Dice Score": "0.764 / 0.757",
        "Loss Function": "Dice + BCE",
        "Key Contribution": "Comparison study with interactive web application"
    },
]

df = pd.DataFrame(papers)

print("=" * 100)
print("LITERATURE SURVEY TABLE")
print("=" * 100)
print(df.to_string(index=False))

# Save
csv_path = os.path.join(Config.PROJECT_ROOT, "docs", "literature_table.csv")
df.to_csv(csv_path, index=False)
print(f"\n✅ Saved to: {csv_path}")

# ============================================================
# PROJECT SUMMARY
# ============================================================
print(f"\n{'='*60}")
print("PROJECT SUMMARY")
print(f"{'='*60}")
print(f"""
Project: Brain Tumor Segmentation using Deep Learning
Dataset: BraTS 2020 (80 patients, 5725 slices)
Modality: FLAIR MRI
Task: Binary segmentation (tumor vs background)
Image Size: 128 x 128

Models:
  U-Net:           31M params, Val Dice 0.8106, Test Dice 0.7643
  Attention U-Net: 35M params, Val Dice 0.8181, Test Dice 0.7572

Training:
  Loss: Dice + Binary Cross Entropy
  Optimizer: Adam (lr=0.0001)
  Hardware: Google Colab T4 GPU
  Time: ~40 min (U-Net), ~57 min (Attention U-Net)

Application: Streamlit web interface for real-time prediction
""")