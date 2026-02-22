
# Brain Tumor Segmentation (BraTS 2020)

This repository contains a CPU-friendly preprocessing pipeline for the **BraTS 2020** dataset and scaffolding for training/evaluating brain tumor segmentation models (e.g., U-Net / Attention U-Net).

## Project structure

```
brain_tumor_segmentation/
  app/                    # (currently scaffolded) app entrypoint
  config/                 # central configuration
  data/                   # raw + processed data (not committed)
  scripts/                # step-by-step pipeline scripts
  src/                    # (currently scaffolded) model/dataset code
  checkpoints/            # saved weights (not committed)
  results/                # figures/outputs (not committed)
```

## Requirements

- Python `>=3.10` (see `setup.py`)

`requirements.txt` is currently empty in this repo. Install the packages used by the scripts manually for now (these are referenced in code):

- `numpy`
- `nibabel`
- `scikit-learn`
- `scikit-image`
- `matplotlib`
- `tqdm`
- `torch`

If you want, tell me your preferred environment manager (pip/conda/poetry) and I can generate a correct `requirements.txt` for your exact setup.

## Setup

Create and activate a virtual environment (example using `venv`):

```bash
python -m venv .venv
```

Then install dependencies (pip example):

```bash
pip install numpy nibabel scikit-learn scikit-image matplotlib tqdm torch
```

## Dataset: download and extract

1. Download the **BraTS 2020 Training** dataset zip.
2. Open `extract_dataset.py` and update `ZIP_FILE_PATH` to the location of your downloaded zip.
3. Run:

```bash
python extract_dataset.py
```

The script extracts to:

```
data/raw/
```

At the end it prints the detected folder path you should copy into your config.

## Configure paths

Open `config/config.py` and confirm that `Config.RAW_DATA_PATH` points to the folder containing the BraTS patient directories (folders that start with `BraTS...`).

You can verify quickly:

```bash
python config/config.py
```

## Pipeline (scripts)

Run these from the project root.

### Step 1: Explore the dataset

```bash
python scripts/01_explore_data.py
```

### Step 2: Visualize modalities + masks

```bash
python scripts/02_visualize_modalities.py
```

Outputs are saved under:

```
results/visualizations/
```

### Step 3: Preprocess into 2D slices

```bash
python scripts/03_preprocess.py
```

This creates `*.npy` arrays under:

```
data/processed/
  train_images.npy
  train_masks.npy
  val_images.npy
  val_masks.npy
  test_images.npy
  test_masks.npy
  metadata.pkl
```

Preprocessing behavior is controlled by `config/config.py`:

- `IMG_SIZE` (default `128`)
- `MODALITY` (default `flair`)
- `USE_SUBSET` / `MAX_PATIENTS` (useful for CPU)
- `EMPTY_SLICE_RATIO` and `MIN_TUMOR_PIXELS` (slice sampling)

### Step 4: Verify preprocessing outputs

```bash
python scripts/04_verify_preprocessing.py
```

This prints dataset stats and saves a visual check image to:

```
results/visualizations/preprocessing_verify.png
```

## Training / evaluation / app

The following files are currently **empty/scaffolded** in this repo:

- `scripts/05_train.py`
- `scripts/06_evaluate.py`
- `scripts/07_compare_models.py`
- `scripts/08_generate_report_table.py`
- `src/*.py` (model/dataset code)
- `app/app.py`

If you want, tell me:

- which model you want first (`UNet` vs `Attention UNet`), and
- whether you want **CPU-only** training or **GPU/Colab**

and I’ll wire up `05_train.py`, `06_evaluate.py`, and a minimal `app/app.py` that loads a checkpoint and runs inference.

## Notes

- This project is configured for Windows CPU runs (`NUM_WORKERS = 0` in `config/config.py`).
- Large folders (`data/`, `checkpoints/`, `results/`) are ignored by `.gitignore`.

