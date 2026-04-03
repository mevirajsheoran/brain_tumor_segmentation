## DOCUMENT 2: Setup Guide & GitHub Instructions

```markdown
# File: docs/SETUP_GUIDE.md

# Setup Guide — For Teammates & GitHub

---

## PART 1: GITHUB SETUP

### Step 1: Create .gitignore

This file tells Git which files to NOT upload.
The dataset is 4GB — NEVER upload that.

Create this file in your project root:

# File: .gitignore

# ============================================================
# DATASET — DO NOT UPLOAD (4GB+)
# ============================================================
data/raw/
data/processed/*.npy
data/processed/*.pkl

# ============================================================
# MODEL CHECKPOINTS — Large files (120MB each)
# Upload to Google Drive or use Git LFS instead
# ============================================================
checkpoints/*.pth
checkpoints/unet/
checkpoints/attention_unet/

# ============================================================
# PYTHON
# ============================================================
__pycache__/
*.pyc
*.pyo
*.egg-info/
*.egg
dist/
build/
.eggs/

# ============================================================
# VIRTUAL ENVIRONMENT — Never upload
# ============================================================
tumor_env/
venv/
env/
.venv/

# ============================================================
# IDE
# ============================================================
.vscode/
.idea/
*.swp
*.swo

# ============================================================
# OS FILES
# ============================================================
.DS_Store
Thumbs.db
desktop.ini

# ============================================================
# JUPYTER
# ============================================================
.ipynb_checkpoints/

# ============================================================
# MISC
# ============================================================
*.log
*.tmp
### Step 2: What TO Upload to GitHub
brain_tumor_segmentation/
├── .gitignore ✅ Upload
├── README.md ✅ Upload
├── setup.py ✅ Upload
├── requirements.txt ✅ Upload
│
├── config/
│ └── config.py ✅ Upload
│
├── src/
│ ├── init.py ✅ Upload
│ ├── unet.py ✅ Upload
│ ├── attention_unet.py ✅ Upload
│ ├── dataset.py ✅ Upload
│ ├── losses.py ✅ Upload
│ └── metrics.py ✅ Upload
│
├── scripts/
│ ├── 01_explore_data.py ✅ Upload
│ ├── 02_visualize_modalities.py ✅ Upload
│ ├── 03_preprocess.py ✅ Upload
│ ├── 04_verify_preprocessing.py ✅ Upload
│ ├── 05_train.py ✅ Upload
│ ├── 06_evaluate.py ✅ Upload
│ ├── 07_compare_models.py ✅ Upload
│ └── 08_generate_report_table.py ✅ Upload
│
├── app/
│ └── app.py ✅ Upload
│
├── notebooks/
│ └── 02_training.ipynb ✅ Upload (Colab notebook)
│
├── results/
│ ├── visualizations/.png ✅ Upload (small images)
│ ├── training_curves/.png ✅ Upload
│ ├── predictions/.png ✅ Upload
│ └── comparison/.png ✅ Upload
│
├── docs/
│ ├── HOW_I_BUILT_THIS.md ✅ Upload
│ ├── SETUP_GUIDE.md ✅ Upload
│ ├── FUTURE_PLANS.md ✅ Upload
│ └── literature_table.csv ✅ Upload
│
├── data/ ❌ DO NOT UPLOAD (4GB dataset)
├── checkpoints/*.pth ❌ DO NOT UPLOAD (120MB models)
├── tumor_env/ ❌ DO NOT UPLOAD (virtual env)
└── pycache/ ❌ DO NOT UPLOAD (cache)


### Step 3: Upload Model Files Separately

Since .pth files are 120MB each (GitHub limit is 100MB):

**Option A: Google Drive (Recommended)**
1. Upload both .pth files to Google Drive
2. Make them shareable (Anyone with link)
3. Add download links to README.md

**Option B: Git LFS**
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
Option C: Hugging Face
Upload models to Hugging Face Hub (free, made for ML models)

Step 4: Create GitHub Repository
Bash

# In your project folder
cd "C:\Users\viraj\Downloads\shrey project\brain_tumor_segmentation"

# Initialize git
git init

# Add all files (respecting .gitignore)
git add .

# Check what will be uploaded
git status

# Commit
git commit -m "Brain Tumor Segmentation - U-Net & Attention U-Net"

# Create repo on GitHub (github.com → New Repository)
# Name: brain-tumor-segmentation
# Keep it public
# Do NOT add README (we have one)

# Connect and push
git remote add origin https://github.com/YOUR_USERNAME/brain-tumor-segmentation.git
git branch -M main
git push -u origin main
Step 5: Create a Good README.md
Replace your README.md with:

text


```markdown
# File: README.md

# 🧠 Brain Tumor Segmentation using Deep Learning

Automatic brain tumor segmentation from MRI scans using U-Net and
Attention U-Net architectures.

![Model Comparison](results/comparison/model_comparison.png)

## 📊 Results

| Metric | U-Net | Attention U-Net |
|--------|-------|-----------------|
| Val Dice | 0.8106 | **0.8181** |
| Test Dice | **0.7643** | 0.7572 |
| Test IoU | **0.6884** | 0.6826 |
| Precision | 0.8675 | **0.8681** |
| Recall | **0.7729** | 0.7725 |

## 🖥️ Live Demo

![App Screenshot](results/predictions/unet_predictions.png)

## 🚀 Quick Setup

### Prerequisites
- Python 3.10+
- ~1GB disk space (without dataset)

### Step 1: Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/brain-tumor-segmentation.git
cd brain-tumor-segmentation
