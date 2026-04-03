# Literature Survey — Brain Tumor Segmentation

## 1. Introduction

Brain tumor segmentation is the task of identifying and delineating
tumor regions from brain MRI scans. This survey covers the key papers
and methods that form the foundation of our project.

The field has evolved from traditional image processing techniques
(thresholding, region growing) to deep learning-based approaches
that achieve significantly higher accuracy and consistency.

---

## 2. Key Papers

### 2.1 U-Net (Ronneberger et al., 2015)

**Title:** U-Net: Convolutional Networks for Biomedical Image Segmentation

**Published:** MICCAI 2015 | Citations: 50,000+

**Problem Solved:** Previous CNNs required thousands of training images.
Medical imaging datasets are small (10-100 patients). U-Net was designed
to work with very few training samples.

**Architecture:**
- Encoder (contracting path): Extracts features using convolutions and
  max pooling. Reduces spatial dimensions while increasing feature channels.
- Decoder (expanding path): Reconstructs spatial resolution using
  transposed convolutions.
- Skip Connections: Direct connections from encoder to decoder at each
  level. This is the KEY innovation — it allows the decoder to use
  high-resolution features from the encoder for precise localization.

**Key Contributions:**
1. Encoder-decoder with skip connections
2. Data augmentation strategies for small datasets
3. Weighted loss function to handle class imbalance
4. Overlap-tile strategy for large images

**Results:** 0.920 Dice on ISBI cell tracking challenge (won first place)

**Why We Used It:** U-Net is the most widely used architecture for
medical image segmentation. It is simple, well-understood, and
consistently performs well across different medical imaging tasks.

**Limitations:**
- Treats all skip connection features equally (no attention)
- 2D processing loses 3D context
- Fixed receptive field size

---

### 2.2 Attention U-Net (Oktay et al., 2018)

**Title:** Attention U-Net: Learning Where to Look for the Pancreas

**Published:** MIDL 2018

**Problem Solved:** In standard U-Net, skip connections pass ALL encoder
features to the decoder, including irrelevant background information.
For small organs/tumors, most of the image is background, making it
harder for the model to focus on the target.

**Key Innovation — Attention Gates:**
Instead of passing raw skip connections, attention gates learn to
filter them. They produce attention weights (0 to 1) for each
spatial location, suppressing irrelevant regions and highlighting
the target.

**How Attention Gates Work:**
Input: gate signal (g) from decoder, skip features (x) from encoder

Transform both: W_g * g and W_x * x (1x1 convolutions)
Add them: combined = ReLU(W_gg + W_xx)
Produce weights: α = Sigmoid(ψ * combined)
Filter: output = x * α

The gate signal from the decoder tells the attention gate "what to
look for" and the skip features provide "where to look." The
attention weights α highlight relevant regions.

**Results:** 0.847 Dice on CT pancreas segmentation

**Why We Used It:** Attention mechanism is a natural extension of U-Net
that should improve performance on small targets like brain tumors
(which occupy <1% of the image).

**Our Finding:** Attention U-Net achieved slightly better validation
Dice (0.818 vs 0.811) but similar test performance (0.757 vs 0.764).
With more training data, the attention mechanism would likely show
larger improvements.

---

### 2.3 nnU-Net (Isensee et al., 2021)

**Title:** nnU-Net: A Self-configuring Method for Deep Learning-based
Biomedical Image Segmentation

**Published:** Nature Methods 2021

**Problem Solved:** Every medical segmentation project requires
extensive manual tuning of preprocessing, architecture, and training
settings. nnU-Net automates ALL of these decisions.

**Key Idea:** The framework analyzes the dataset properties (image
size, spacing, class distribution) and automatically configures:
- Preprocessing pipeline
- Network architecture (2D, 3D, or cascade)
- Training schedule
- Post-processing

**Results:** 0.885 Dice on BraTS 2020 (state-of-the-art)

**Why It Matters:** nnU-Net proves that systematic preprocessing and
training configuration are as important as the architecture itself.
Our project follows similar preprocessing principles (normalization,
resampling, class balancing).

**Why We Didn't Use It:** nnU-Net is a complete framework, not a
single model. Our project goal was to understand and implement
architectures from scratch, not use an automated pipeline.

---

### 2.4 BraTS Challenge (Menze et al., 2015)

**Title:** The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)

**Published:** IEEE TMI 2015

**What It Is:** The Brain Tumor Segmentation Challenge is an annual
competition where teams worldwide develop algorithms to segment
brain tumors from MRI scans. It provides:
- Standardized dataset with expert annotations
- Consistent evaluation metrics
- Public leaderboard for comparison

**Dataset Details:**
- Multi-institutional (19 different hospitals/centers)
- 4 MRI modalities: T1, T1ce, T2, FLAIR
- 3 tumor regions: Necrotic core, edema, enhancing tumor
- Expert annotations verified by neuroradiologists

**Evaluation Metrics:**
- Dice Similarity Coefficient (primary)
- Hausdorff Distance (boundary accuracy)
- Sensitivity and Specificity

**Why We Used BraTS 2020:** It is the gold standard dataset for brain
tumor segmentation. Using it makes our results directly comparable
to published research.

---

### 2.5 Myronenko (2019) — BraTS 2018 Winner

**Title:** 3D MRI Brain Tumor Segmentation Using Autoencoder
Regularization

**Published:** BrainLes Workshop, MICCAI 2018

**Key Innovation:** Combined segmentation with a variational
autoencoder (VAE) that reconstructs the input image. The VAE
acts as a regularizer, preventing overfitting and improving
feature learning.

**Architecture:**
- 3D encoder-decoder (processes full volume)
- Additional VAE branch for regularization
- Multi-scale loss function

**Results:** 0.884 Dice — Won BraTS 2018 challenge

**What We Learned:** 3D processing captures inter-slice relationships
that 2D approaches miss. VAE regularization is an elegant way to
improve generalization with limited data.

---

## 3. Comparison Table

| Paper | Year | Model | Dataset | Dice | Loss | Key Idea |
|-------|------|-------|---------|------|------|----------|
| Ronneberger | 2015 | U-Net | ISBI Cell | 0.920 | CE + Weights | Skip connections |
| Oktay | 2018 | Att U-Net | CT Abdomen | 0.847 | Dice | Attention gates |
| Myronenko | 2019 | 3D VAE | BraTS 2018 | 0.884 | Dice+L2+KL | VAE regularization |
| Isensee | 2021 | nnU-Net | BraTS 2020 | 0.885 | Dice+CE | Auto-configuration |
| **Ours** | **2024** | **U-Net + Att** | **BraTS 2020** | **0.764** | **Dice+BCE** | **Comparison + App** |

---

## 4. Research Gap Identified

After reviewing the literature, we identified these gaps:

1. **Accessibility:** Most papers focus on achieving maximum accuracy
   but don't provide user-friendly tools for non-ML practitioners.
   Our project includes a Streamlit web application.

2. **Systematic Comparison:** Many papers propose a single architecture
   without comparing alternatives on the same dataset with identical
   preprocessing. We compare U-Net and Attention U-Net under
   identical conditions.

3. **2D Feasibility:** Most recent work uses 3D architectures. We
   demonstrate that 2D approaches still achieve reasonable
   performance (Dice 0.76) with significantly lower computational
   requirements.

---

## 5. Metrics Explained

### Dice Similarity Coefficient (DSC)

The primary metric for segmentation quality.
 
The gate signal from the decoder tells the attention gate "what to
look for" and the skip features provide "where to look." The
attention weights α highlight relevant regions.

**Results:** 0.847 Dice on CT pancreas segmentation

**Why We Used It:** Attention mechanism is a natural extension of U-Net
that should improve performance on small targets like brain tumors
(which occupy <1% of the image).

**Our Finding:** Attention U-Net achieved slightly better validation
Dice (0.818 vs 0.811) but similar test performance (0.757 vs 0.764).
With more training data, the attention mechanism would likely show
larger improvements.

---

### 2.3 nnU-Net (Isensee et al., 2021)

**Title:** nnU-Net: A Self-configuring Method for Deep Learning-based
Biomedical Image Segmentation

**Published:** Nature Methods 2021

**Problem Solved:** Every medical segmentation project requires
extensive manual tuning of preprocessing, architecture, and training
settings. nnU-Net automates ALL of these decisions.

**Key Idea:** The framework analyzes the dataset properties (image
size, spacing, class distribution) and automatically configures:
- Preprocessing pipeline
- Network architecture (2D, 3D, or cascade)
- Training schedule
- Post-processing

**Results:** 0.885 Dice on BraTS 2020 (state-of-the-art)

**Why It Matters:** nnU-Net proves that systematic preprocessing and
training configuration are as important as the architecture itself.
Our project follows similar preprocessing principles (normalization,
resampling, class balancing).

**Why We Didn't Use It:** nnU-Net is a complete framework, not a
single model. Our project goal was to understand and implement
architectures from scratch, not use an automated pipeline.

---

### 2.4 BraTS Challenge (Menze et al., 2015)

**Title:** The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)

**Published:** IEEE TMI 2015

**What It Is:** The Brain Tumor Segmentation Challenge is an annual
competition where teams worldwide develop algorithms to segment
brain tumors from MRI scans. It provides:
- Standardized dataset with expert annotations
- Consistent evaluation metrics
- Public leaderboard for comparison

**Dataset Details:**
- Multi-institutional (19 different hospitals/centers)
- 4 MRI modalities: T1, T1ce, T2, FLAIR
- 3 tumor regions: Necrotic core, edema, enhancing tumor
- Expert annotations verified by neuroradiologists

**Evaluation Metrics:**
- Dice Similarity Coefficient (primary)
- Hausdorff Distance (boundary accuracy)
- Sensitivity and Specificity

**Why We Used BraTS 2020:** It is the gold standard dataset for brain
tumor segmentation. Using it makes our results directly comparable
to published research.

---

### 2.5 Myronenko (2019) — BraTS 2018 Winner

**Title:** 3D MRI Brain Tumor Segmentation Using Autoencoder
Regularization

**Published:** BrainLes Workshop, MICCAI 2018

**Key Innovation:** Combined segmentation with a variational
autoencoder (VAE) that reconstructs the input image. The VAE
acts as a regularizer, preventing overfitting and improving
feature learning.

**Architecture:**
- 3D encoder-decoder (processes full volume)
- Additional VAE branch for regularization
- Multi-scale loss function

**Results:** 0.884 Dice — Won BraTS 2018 challenge

**What We Learned:** 3D processing captures inter-slice relationships
that 2D approaches miss. VAE regularization is an elegant way to
improve generalization with limited data.

---

## 3. Comparison Table

| Paper | Year | Model | Dataset | Dice | Loss | Key Idea |
|-------|------|-------|---------|------|------|----------|
| Ronneberger | 2015 | U-Net | ISBI Cell | 0.920 | CE + Weights | Skip connections |
| Oktay | 2018 | Att U-Net | CT Abdomen | 0.847 | Dice | Attention gates |
| Myronenko | 2019 | 3D VAE | BraTS 2018 | 0.884 | Dice+L2+KL | VAE regularization |
| Isensee | 2021 | nnU-Net | BraTS 2020 | 0.885 | Dice+CE | Auto-configuration |
| **Ours** | **2024** | **U-Net + Att** | **BraTS 2020** | **0.764** | **Dice+BCE** | **Comparison + App** |

---

## 4. Research Gap Identified

After reviewing the literature, we identified these gaps:

1. **Accessibility:** Most papers focus on achieving maximum accuracy
   but don't provide user-friendly tools for non-ML practitioners.
   Our project includes a Streamlit web application.

2. **Systematic Comparison:** Many papers propose a single architecture
   without comparing alternatives on the same dataset with identical
   preprocessing. We compare U-Net and Attention U-Net under
   identical conditions.

3. **2D Feasibility:** Most recent work uses 3D architectures. We
   demonstrate that 2D approaches still achieve reasonable
   performance (Dice 0.76) with significantly lower computational
   requirements.

---

## 5. Metrics Explained

### Dice Similarity Coefficient (DSC)

The primary metric for segmentation quality.
- Range: 0 (no overlap) to 1 (perfect overlap)
- Equivalent to F1-score for binary classification
- Handles class imbalance better than accuracy

**Example:**
- Prediction has 100 tumor pixels
- Ground truth has 120 tumor pixels
- 80 pixels overlap
- Dice = (2 × 80) / (100 + 120) = 160/220 = 0.727

### Intersection over Union (IoU)
- Always lower than Dice for the same prediction
- Relationship: Dice = 2×IoU / (1+IoU)
- More intuitive: "what fraction of the combined area overlaps?"

### Precision

"Of everything I predicted as tumor, how much is actually tumor?"
High precision = few false alarms.

### Recall (Sensitivity)
"Of all actual tumor, how much did I detect?"
High recall = few missed tumors.

In medical imaging, recall is often prioritized because missing
a tumor (false negative) is more dangerous than a false alarm
(false positive).

---

## 6. References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net:
   Convolutional Networks for Biomedical Image Segmentation.
   MICCAI, 234-241.

2. Oktay, O., Schlemper, J., Folgoc, L.L., et al. (2018).
   Attention U-Net: Learning Where to Look for the Pancreas.
   Medical Imaging with Deep Learning (MIDL).

3. Isensee, F., Jaeger, P.F., Kohl, S.A., et al. (2021).
   nnU-Net: a self-configuring method for deep learning-based
   biomedical image segmentation. Nature Methods, 18(2), 203-211.

4. Menze, B.H., Jakab, A., Bauer, S., et al. (2015). The
   Multimodal Brain Tumor Image Segmentation Benchmark (BRATS).
   IEEE Transactions on Medical Imaging, 34(10), 1993-2024.

5. Myronenko, A. (2019). 3D MRI Brain Tumor Segmentation Using
   Autoencoder Regularization. BrainLes Workshop, MICCAI 2018,
   LNCS 11384, 311-320.