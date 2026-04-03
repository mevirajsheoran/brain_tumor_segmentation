# File: app/app.py
# Location: brain_tumor_segmentation\app\app.py

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io
import os
import sys
import tempfile
import json

# ============================================================
# Add project root to path so we can import src/
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.unet import UNet
from src.attention_unet import AttentionUNet

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Brain Tumor Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS FOR BEAUTIFUL UI
# ============================================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0a2e 0%, #1a1a4e 50%, #0d0d3d 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 3rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #1e1e4a 0%, #2a2a5e 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
    }
    
    .metric-card h3 {
        color: #667eea;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-card .value {
        color: white;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .metric-card .value.positive {
        color: #4ade80;
    }
    
    .metric-card .value.warning {
        color: #fbbf24;
    }
    
    .metric-card .value.danger {
        color: #f87171;
    }
    
    /* Result section */
    .result-section {
        background: linear-gradient(135deg, #1a1a4e 0%, #252560 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 5px 20px rgba(0,0,0,0.3);
    }
    
    .result-section h2 {
        color: #667eea;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
        padding-bottom: 0.5rem;
    }
    
    /* Status badges */
    .badge-detected {
        background: linear-gradient(135deg, #059669, #10b981);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
    }
    
    .badge-clear {
        background: linear-gradient(135deg, #2563eb, #3b82f6);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
    }
    
    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a4e 0%, #0d0d3d 100%);
    }
    
    /* Upload area */
    .upload-area {
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 15px;
        padding: 3rem 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        margin: 1rem 0;
    }
    
    /* Info box */
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 1rem 1.5rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
        color: rgba(255,255,255,0.8);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: rgba(255,255,255,0.4);
        border-top: 1px solid rgba(255,255,255,0.1);
        margin-top: 3rem;
    }
    
    /* Fix text colors */
    .stMarkdown, p, span, label {
        color: rgba(255,255,255,0.85) !important;
    }
    
    h1, h2, h3 {
        color: white !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 8px;
        color: white;
        padding: 0.5rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource
def load_models():
    """Load both trained models — cached so only loads once"""
    device = torch.device("cpu")
    models = {}
    
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    
    if not os.path.exists(checkpoint_dir):
        return None, "Checkpoints folder not found!"
    
    pth_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    
    if len(pth_files) == 0:
        return None, "No .pth model files found in checkpoints/"
    
    for f in pth_files:
        filepath = os.path.join(checkpoint_dir, f)
        try:
            # =============================================
            # THIS IS THE FIX — weights_only=False
            # =============================================
            ckpt = torch.load(filepath, map_location=device, weights_only=False)
            
            model_name = ckpt.get('model_name', '')
            features = ckpt.get('features', [64, 128, 256, 512])
            
            if 'attention' in f.lower() or 'Attention' in model_name:
                model = AttentionUNet(
                    in_channels=ckpt.get('in_channels', 1),
                    out_channels=ckpt.get('out_channels', 1),
                    features=features
                )
                model.load_state_dict(ckpt['model_state_dict'])
                model.eval()
                models['Attention U-Net'] = {
                    'model': model,
                    'dice': ckpt.get('test_dice', ckpt.get('best_val_dice', 'N/A')),
                    'iou': ckpt.get('test_iou', 'N/A'),
                    'precision': ckpt.get('test_precision', 'N/A'),
                    'recall': ckpt.get('test_recall', 'N/A'),
                }
            else:
                model = UNet(
                    in_channels=ckpt.get('in_channels', 1),
                    out_channels=ckpt.get('out_channels', 1),
                    features=features
                )
                model.load_state_dict(ckpt['model_state_dict'])
                model.eval()
                models['U-Net'] = {
                    'model': model,
                    'dice': ckpt.get('test_dice', ckpt.get('best_val_dice', 'N/A')),
                    'iou': ckpt.get('test_iou', 'N/A'),
                    'precision': ckpt.get('test_precision', 'N/A'),
                    'recall': ckpt.get('test_recall', 'N/A'),
                }
        except Exception as e:
            st.warning(f"Could not load {f}: {e}")
    
    if len(models) == 0:
        return None, "Failed to load any models!"
    
    return models, None


# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict(model, image_array, threshold=0.5):
    """Run prediction on a single 2D image"""
    from skimage.transform import resize
    
    # Ensure grayscale and resize to 128x128
    if len(image_array.shape) == 3:
        image_array = np.mean(image_array, axis=2)
    
    original_shape = image_array.shape
    
    # Normalize
    img = image_array.astype(np.float32)
    if img.max() > 1.0:
        img = img / img.max()
    
    # Resize to model input size
    img_resized = resize(img, (128, 128), preserve_range=True,
                         anti_aliasing=True).astype(np.float32)
    
    # To tensor
    input_tensor = torch.FloatTensor(img_resized).unsqueeze(0).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        prob_map = torch.sigmoid(output).numpy().squeeze()
        binary_mask = (prob_map > threshold).astype(np.float32)
    
    # Resize mask back to original size
    if original_shape != (128, 128):
        prob_map_full = resize(prob_map, original_shape, preserve_range=True,
                               anti_aliasing=True)
        binary_mask_full = resize(binary_mask, original_shape, preserve_range=True,
                                  order=0, anti_aliasing=False)
        binary_mask_full = (binary_mask_full > 0.5).astype(np.float32)
    else:
        prob_map_full = prob_map
        binary_mask_full = binary_mask
    
    # Calculate metrics
    tumor_pixels = int(binary_mask.sum())
    total_pixels = binary_mask.size
    tumor_percentage = (tumor_pixels / total_pixels) * 100
    
    return {
        'prob_map': prob_map,
        'binary_mask': binary_mask,
        'prob_map_full': prob_map_full,
        'binary_mask_full': binary_mask_full,
        'tumor_pixels': tumor_pixels,
        'total_pixels': total_pixels,
        'tumor_percentage': tumor_percentage,
        'img_resized': img_resized,
    }


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================
def create_result_figure(image, prob_map, binary_mask, title=""):
    """Create a clean 4-panel result figure"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor('#0a0a2e')
    
    for ax in axes:
        ax.set_facecolor('#0a0a2e')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
    
    # 1. Original MRI
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Input MRI', fontsize=13, color='white', fontweight='bold')
    axes[0].axis('off')
    
    # 2. Probability Map
    im = axes[1].imshow(prob_map, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Probability Map', fontsize=13, color='white', fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. Binary Mask
    axes[2].imshow(binary_mask, cmap='gray')
    axes[2].set_title('Segmentation Mask', fontsize=13, color='white', fontweight='bold')
    axes[2].axis('off')
    
    # 4. Overlay
    axes[3].imshow(image, cmap='gray')
    mask_overlay = np.zeros((*binary_mask.shape, 4))
    mask_overlay[binary_mask > 0] = [1, 0, 0, 0.45]
    axes[3].imshow(mask_overlay)
    axes[3].set_title('Overlay', fontsize=13, color='white', fontweight='bold')
    axes[3].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=15, color='white', fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig


def create_comparison_figure(image, results_dict):
    """Create comparison figure for multiple models"""
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, 2 + n_models, figsize=(6 * (2 + n_models), 5))
    fig.patch.set_facecolor('#0a0a2e')
    
    for ax in axes:
        ax.set_facecolor('#0a0a2e')
        ax.axis('off')
    
    # Original
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Input MRI', fontsize=13, color='white', fontweight='bold')
    
    # Each model's prediction
    for idx, (model_name, result) in enumerate(results_dict.items()):
        ax = axes[1 + idx]
        ax.imshow(image, cmap='gray')
        mask_overlay = np.zeros((*result['binary_mask'].shape, 4))
        mask_overlay[result['binary_mask'] > 0] = [1, 0, 0, 0.45]
        ax.imshow(mask_overlay)
        ax.set_title(f'{model_name}\n{result["tumor_percentage"]:.1f}% tumor',
                     fontsize=12, color='white', fontweight='bold')
    
    # Probability comparison (last panel)
    axes[-1].imshow(list(results_dict.values())[0]['prob_map'], cmap='hot')
    axes[-1].set_title('Probability Map', fontsize=13, color='white', fontweight='bold')
    
    plt.tight_layout()
    return fig


# ============================================================
# MAIN APP
# ============================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>🧠 Brain Tumor Segmentation</h1>
    <p>Deep Learning-based MRI Brain Tumor Detection using U-Net & Attention U-Net</p>
</div>
""", unsafe_allow_html=True)

# Load models
models, error = load_models()

if error:
    st.error(f"❌ {error}")
    st.info("Make sure your .pth model files are in the 'checkpoints/' folder")
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")
    
    # Model selection
    available_models = list(models.keys())
    selected_model = st.selectbox(
        "🤖 Select Model",
        available_models,
        index=0
    )
    
    # Threshold
    threshold = st.slider(
        "🎯 Segmentation Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Lower = more sensitive (detects more), Higher = more specific"
    )
    
    # Overlay opacity
    opacity = st.slider(
        "👁️ Overlay Opacity",
        min_value=0.1,
        max_value=1.0,
        value=0.4,
        step=0.1
    )
    
    st.markdown("---")
    
    # Model info
    st.markdown("## 📊 Model Info")
    
    model_info = models[selected_model]
    
    dice_val = model_info['dice']
    if isinstance(dice_val, float):
        dice_display = f"{dice_val:.4f}"
    else:
        dice_display = str(dice_val)
    
    st.markdown(f"""
    - **Model:** {selected_model}
    - **Test Dice:** {dice_display}
    - **Architecture:** Encoder-Decoder
    - **Input Size:** 128 × 128
    - **Dataset:** BraTS 2020
    """)
    
    st.markdown("---")
    
    # Compare models toggle
    compare_mode = False
    if len(available_models) > 1:
        compare_mode = st.checkbox("🔄 Compare All Models", value=False)
    
    st.markdown("---")
    st.markdown("""
    ### 📚 About
    - **Task:** Binary segmentation
    - **Input:** FLAIR MRI
    - **Output:** Tumor mask
    - **Framework:** PyTorch
    """)

# ============================================================
# MAIN CONTENT — TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Analyze MRI", 
    "📊 Model Performance",
    "📖 About Project",
    "❓ How To Use"
])

# ============================================================
# TAB 1: ANALYZE MRI
# ============================================================
with tab1:
    st.markdown("## 📤 Upload MRI Scan")
    
    col_upload1, col_upload2 = st.columns([2, 1])
    
    with col_upload1:
        uploaded_file = st.file_uploader(
            "Choose an MRI image",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload a brain MRI slice image. Grayscale works best."
        )
    
    with col_upload2:
        st.markdown("""
        <div class="info-box">
            <strong>Supported Formats:</strong><br>
            PNG, JPG, JPEG, BMP, TIFF<br><br>
            <strong>Best Results With:</strong><br>
            • FLAIR MRI sequences<br>
            • Axial view slices<br>
            • Grayscale images
        </div>
        """, unsafe_allow_html=True)
    
    # Also allow loading from test dataset
    use_sample = st.checkbox("📂 Or use a sample from test dataset")
    
    image_to_process = None
    
    if use_sample:
        test_path = os.path.join(PROJECT_ROOT, "data", "processed", "test_images.npy")
        if os.path.exists(test_path):
            test_images = np.load(test_path)
            test_masks = np.load(os.path.join(PROJECT_ROOT, "data", "processed", "test_masks.npy"))
            
            # Find slices with tumor
            tumor_sums = test_masks.reshape(len(test_masks), -1).sum(axis=1)
            tumor_indices = np.where(tumor_sums > 100)[0]
            
            if len(tumor_indices) > 0:
                sample_idx = st.slider(
                    "Select sample (tumor slices only)",
                    0, len(tumor_indices) - 1, 0
                )
                actual_idx = tumor_indices[sample_idx]
                image_to_process = test_images[actual_idx]
                ground_truth = test_masks[actual_idx]
                st.success(f"Loaded test sample #{actual_idx} "
                          f"({int(ground_truth.sum())} tumor pixels)")
            else:
                st.warning("No tumor slices found in test set")
        else:
            st.warning(f"Test data not found at: {test_path}")
            st.info("Make sure data/processed/test_images.npy exists")
    
    elif uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_to_process = np.array(image.convert('L')).astype(np.float32)
        if image_to_process.max() > 1:
            image_to_process = image_to_process / 255.0
        ground_truth = None
        st.success(f"Image loaded: {image_to_process.shape}")
    
    # ============================================================
    # PROCESS AND SHOW RESULTS
    # ============================================================
    if image_to_process is not None:
        st.markdown("---")
        
        if compare_mode and len(available_models) > 1:
            # Compare all models
            st.markdown("## 🔄 Model Comparison Results")
            
            all_results = {}
            for model_name in available_models:
                model_obj = models[model_name]['model']
                result = predict(model_obj, image_to_process, threshold)
                all_results[model_name] = result
            
            # Metrics comparison
            metric_cols = st.columns(len(available_models))
            for idx, (model_name, result) in enumerate(all_results.items()):
                with metric_cols[idx]:
                    detected = result['tumor_pixels'] > 0
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{model_name}</h3>
                        <div class="value {'positive' if detected else ''}">
                            {'🔴 Tumor Detected' if detected else '🟢 No Tumor'}
                        </div>
                        <p>Tumor Area: {result['tumor_percentage']:.2f}%</p>
                        <p>Tumor Pixels: {result['tumor_pixels']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Comparison figure
            fig = create_comparison_figure(
                all_results[available_models[0]]['img_resized'],
                all_results
            )
            st.pyplot(fig)
            plt.close(fig)
            
            # Individual results
            for model_name, result in all_results.items():
                with st.expander(f"📋 {model_name} Detailed View"):
                    fig = create_result_figure(
                        result['img_resized'],
                        result['prob_map'],
                        result['binary_mask'],
                        title=model_name
                    )
                    st.pyplot(fig)
                    plt.close(fig)
        
        else:
            # Single model prediction
            st.markdown(f"## 🔬 Results — {selected_model}")
            
            model_obj = models[selected_model]['model']
            result = predict(model_obj, image_to_process, threshold)
            
            # ---- Metrics Cards ----
            col1, col2, col3, col4 = st.columns(4)
            
            detected = result['tumor_pixels'] > 0
            
            with col1:
                badge_class = "badge-detected" if detected else "badge-clear"
                badge_text = "🔴 Tumor Detected" if detected else "🟢 No Tumor"
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Status</h3>
                    <span class="{badge_class}">{badge_text}</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                pct = result['tumor_percentage']
                color_class = 'danger' if pct > 5 else 'warning' if pct > 1 else 'positive'
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Tumor Area</h3>
                    <div class="value {color_class}">{pct:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Tumor Pixels</h3>
                    <div class="value">{result['tumor_pixels']:,}</div>
                    <p>of {result['total_pixels']:,} total</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                confidence = result['prob_map'].max() * 100 if detected else 0
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Max Confidence</h3>
                    <div class="value">{confidence:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # ---- Visualization ----
            fig = create_result_figure(
                result['img_resized'],
                result['prob_map'],
                result['binary_mask'],
                title=f"{selected_model} Prediction"
            )
            st.pyplot(fig)
            plt.close(fig)
            
            # ---- Ground Truth Comparison (if using test sample) ----
            if use_sample and 'ground_truth' in dir() and ground_truth is not None:
                st.markdown("### 📏 Comparison with Ground Truth")
                
                from skimage.transform import resize as sk_resize
                gt_resized = sk_resize(ground_truth, (128, 128), order=0,
                                       preserve_range=True).astype(np.float32)
                gt_resized = (gt_resized > 0.5).astype(np.float32)
                
                pred_flat = result['binary_mask'].flatten()
                gt_flat = gt_resized.flatten()
                
                intersection = (pred_flat * gt_flat).sum()
                dice = (2 * intersection + 1e-6) / (pred_flat.sum() + gt_flat.sum() + 1e-6)
                union = pred_flat.sum() + gt_flat.sum() - intersection
                iou = (intersection + 1e-6) / (union + 1e-6)
                
                tp = (pred_flat * gt_flat).sum()
                fp = (pred_flat * (1 - gt_flat)).sum()
                fn = ((1 - pred_flat) * gt_flat).sum()
                precision = (tp + 1e-6) / (tp + fp + 1e-6)
                recall = (tp + 1e-6) / (tp + fn + 1e-6)
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Dice Score", f"{dice:.4f}")
                m2.metric("IoU Score", f"{iou:.4f}")
                m3.metric("Precision", f"{precision:.4f}")
                m4.metric("Recall", f"{recall:.4f}")
                
                # Ground truth overlay
                fig_gt, axes_gt = plt.subplots(1, 3, figsize=(15, 5))
                fig_gt.patch.set_facecolor('#0a0a2e')
                
                for ax in axes_gt:
                    ax.axis('off')
                    ax.set_facecolor('#0a0a2e')
                
                axes_gt[0].imshow(gt_resized, cmap='gray')
                axes_gt[0].set_title('Ground Truth', color='white', fontsize=13)
                
                axes_gt[1].imshow(result['binary_mask'], cmap='gray')
                axes_gt[1].set_title('Prediction', color='white', fontsize=13)
                
                # Overlap visualization
                overlap_img = np.zeros((*gt_resized.shape, 3))
                overlap_img[gt_resized > 0] = [0, 1, 0]         # Green = GT
                overlap_img[result['binary_mask'] > 0] = [1, 0, 0]  # Red = Pred
                both = (gt_resized > 0) & (result['binary_mask'] > 0)
                overlap_img[both] = [1, 1, 0]                    # Yellow = Both
                
                axes_gt[2].imshow(result['img_resized'], cmap='gray')
                axes_gt[2].imshow(overlap_img, alpha=0.5)
                axes_gt[2].set_title('Green=GT  Red=Pred  Yellow=Both',
                                     color='white', fontsize=11)
                
                plt.tight_layout()
                st.pyplot(fig_gt)
                plt.close(fig_gt)

# ============================================================
# TAB 2: MODEL PERFORMANCE
# ============================================================
with tab2:
    st.markdown("## 📊 Model Performance Summary")
    
    # Results table
    st.markdown("""
    <div class="result-section">
        <h2>Test Set Results — BraTS 2020 Dataset</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Load results from json if available
    results_json = os.path.join(PROJECT_ROOT, "checkpoints", "results_summary.json")
    
    if os.path.exists(results_json):
        with open(results_json, 'r') as f:
            saved_results = json.load(f)
        
        col1, col2 = st.columns(2)
        
        for idx, (model_name, res) in enumerate(saved_results.items()):
            with [col1, col2][idx]:
                display_name = "U-Net" if "UNet" == model_name else "Attention U-Net"
                st.markdown(f"### {display_name}")
                st.metric("Dice Score", f"{res.get('test_dice', 'N/A'):.4f}" 
                          if isinstance(res.get('test_dice'), float) else 'N/A')
                st.metric("IoU Score", f"{res.get('test_iou', 'N/A'):.4f}"
                          if isinstance(res.get('test_iou'), float) else 'N/A')
                st.metric("Precision", f"{res.get('test_precision', 'N/A'):.4f}"
                          if isinstance(res.get('test_precision'), float) else 'N/A')
                st.metric("Recall", f"{res.get('test_recall', 'N/A'):.4f}"
                          if isinstance(res.get('test_recall'), float) else 'N/A')
                st.metric("Training Epochs", res.get('epochs_trained', 'N/A'))
                st.metric("Training Time", f"{res.get('training_time_min', 'N/A')} min")
    else:
        # Show from loaded model info
        for model_name, info in models.items():
            st.markdown(f"### {model_name}")
            if isinstance(info['dice'], float):
                st.metric("Test Dice Score", f"{info['dice']:.4f}")
            if isinstance(info.get('iou'), float):
                st.metric("Test IoU Score", f"{info['iou']:.4f}")
    
    # Show saved images if they exist
    st.markdown("---")
    st.markdown("### 📈 Training Curves & Comparisons")
    
    image_locations = [
        os.path.join(PROJECT_ROOT, "checkpoints"),
        os.path.join(PROJECT_ROOT, "results", "comparison"),
        os.path.join(PROJECT_ROOT, "results"),
    ]
    
    for loc in image_locations:
        if os.path.exists(loc):
            for f in os.listdir(loc):
              if f.endswith('.png'):
                  img_path = os.path.join(loc, f)
                  try:
                      from PIL import Image as PILImage
                      test_img = PILImage.open(img_path)
                      test_img.verify()
                      st.image(img_path, caption=f, width="stretch")
                  except Exception:
                      pass

# ============================================================
# TAB 3: ABOUT
# ============================================================
with tab3:
    st.markdown("""
    ## 📖 About This Project
    
    ### Problem Statement
    Brain tumors are among the most serious medical conditions requiring accurate 
    and timely diagnosis. Manual segmentation of brain tumors from MRI scans is 
    time-consuming and subject to inter-observer variability. This project implements 
    deep learning models to automatically segment brain tumors from MRI scans.
    
    ### Dataset
    - **Name:** BraTS 2020 (Brain Tumor Segmentation Challenge)
    - **Patients:** 369 (80 used for training)
    - **MRI Modalities:** FLAIR, T1, T1ce, T2
    - **Used Modality:** FLAIR
    - **Task:** Binary segmentation (tumor vs. background)
    
    ### Models Implemented
    
    #### 1. U-Net
    - Original architecture by Ronneberger et al. (2015)
    - Encoder-decoder with skip connections
    - Parameters: ~31M
    
    #### 2. Attention U-Net
    - Enhanced U-Net with attention gates (Oktay et al., 2018)
    - Focuses on relevant regions automatically
    - Parameters: ~35M
    
    ### Technical Details
    - **Framework:** PyTorch
    - **Loss Function:** Dice Loss + Binary Cross Entropy
    - **Optimizer:** Adam (lr=0.0001)
    - **Image Size:** 128 × 128
    - **Preprocessing:** Intensity normalization, resize, slice extraction
    - **Data Split:** Patient-wise (70/15/15)
    
    ### Literature Survey
    
    | Paper | Model | Dataset | Dice Score |
    |-------|-------|---------|------------|
    | Ronneberger et al. (2015) | U-Net | ISBI Cell | 0.920 |
    | Oktay et al. (2018) | Attention U-Net | CT Abdomen | 0.847 |
    | Isensee et al. (2021) | nnU-Net | BraTS 2020 | 0.885 |
    | Myronenko (2019) | 3D Enc-Dec + VAE | BraTS 2018 | 0.884 |
    | **This Project** | **U-Net + Att U-Net** | **BraTS 2020** | **0.764 / 0.757** |
    """)

# ============================================================
# TAB 4: HOW TO USE
# ============================================================
with tab4:
    st.markdown("""
    ## ❓ How To Use This Application
    
    ### Step 1: Select Model
    Choose between **U-Net** and **Attention U-Net** from the sidebar.
    
    ### Step 2: Upload Image
    - Go to the **"🔬 Analyze MRI"** tab
    - Upload a brain MRI image (PNG/JPG)
    - OR check **"Use sample from test dataset"** to try with real data
    
    ### Step 3: Adjust Settings
    - **Threshold:** Lower values detect more (but may include false positives)
    - **Opacity:** Controls overlay transparency
    
    ### Step 4: View Results
    - **Status:** Whether tumor is detected
    - **Tumor Area:** Percentage of image classified as tumor
    - **Probability Map:** Confidence of prediction at each pixel
    - **Segmentation Mask:** Binary tumor/non-tumor prediction
    - **Overlay:** Tumor highlighted on original MRI
    
    ### Step 5: Compare Models
    - Check **"🔄 Compare All Models"** in sidebar
    - See side-by-side predictions from both models
    
    ### Understanding Results
    
    | Metric | Meaning | Good Value |
    |--------|---------|------------|
    | Dice Score | Overlap between prediction and truth | > 0.75 |
    | IoU | Intersection over Union | > 0.65 |
    | Precision | How many predicted tumors are real | > 0.80 |
    | Recall | How many real tumors were found | > 0.75 |
    
    ### ⚠️ Disclaimer
    This is a **research project** and should NOT be used for clinical diagnosis. 
    Always consult qualified medical professionals for medical decisions.
    """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    <p>🧠 Brain Tumor Segmentation System | Built with PyTorch & Streamlit</p>
    <p>Dataset: BraTS 2020 | Models: U-Net & Attention U-Net</p>
</div>
""", unsafe_allow_html=True)