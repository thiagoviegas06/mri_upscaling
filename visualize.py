import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# Import from your existing files
from preprocessing import load_pair_resample_normalize
from test import load_model, predict_volume

def visualize_random_pairs(lf_path, hf_path, save_dir="visualization"):
    """
    1. Polls 5 random 33-66 percentile slices.
    2. Plots LF slices, HF slices, and a line plot of their noise/intensity 
       distributions for each slice individually, then saves them.
    """
    os.makedirs(save_dir, exist_ok=True)
    sample_name = Path(lf_path).name.replace('_lowfield.nii', '').replace('.nii', '')
    
    # Load and preprocess volumes
    lf_vol, hf_vol = load_pair_resample_normalize(lf_path, hf_path)
    
    # Calculate 33rd and 66th percentile indices along the Z-axis (depth)
    z_dim = lf_vol.shape[2]
    z_min, z_max = int(z_dim * 0.33), int(z_dim * 0.66)
    
    # Sample 5 random slices
    slice_idxs = random.sample(range(z_min, z_max), 5)
    
    for z in slice_idxs:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'{sample_name} - LF/HF Pair (Slice {z})', fontsize=14)
        
        # Left: Low Field
        axes[0].imshow(lf_vol[:, :, z], cmap='gray')
        axes[0].set_title(f'LF Slice {z}')
        axes[0].axis('off')
        
        # Middle: High Field (Ground Truth)
        axes[1].imshow(hf_vol[:, :, z], cmap='gray')
        axes[1].set_title(f'HF Slice {z}')
        axes[1].axis('off')
        
        # Right: Noise / Intensity Distribution as Line Plots
        lf_flat = lf_vol[:, :, z].flatten()
        hf_flat = hf_vol[:, :, z].flatten()
        
        # Calculate histogram data
        lf_counts, lf_bins = np.histogram(lf_flat, bins=50, density=True)
        hf_counts, hf_bins = np.histogram(hf_flat, bins=50, density=True)
        
        # Calculate bin centers for plotting
        lf_centers = (lf_bins[:-1] + lf_bins[1:]) / 2
        hf_centers = (hf_bins[:-1] + hf_bins[1:]) / 2
        
        # Plot as lines
        axes[2].plot(lf_centers, lf_counts, color='blue', label='LF', linewidth=2)
        axes[2].plot(hf_centers, hf_counts, color='orange', label='HF', linewidth=2)
        
        # Add a slight fill to make it pop visually (optional but recommended)
        axes[2].fill_between(lf_centers, lf_counts, alpha=0.2, color='blue')
        axes[2].fill_between(hf_centers, hf_counts, alpha=0.2, color='orange')
        
        axes[2].set_title('Intensity / Noise Distribution')
        axes[2].set_xlabel('Voxel Intensity')
        axes[2].set_ylabel('Density')
        axes[2].legend()
        
        plt.tight_layout()
        
        # Save figure before showing
        save_path = os.path.join(save_dir, f"{sample_name}_lf_hf_pair_slice_{z}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved: {save_path}")
        
        # Display the figure
        plt.show()

def visualize_model_predictions(lf_path, hf_path, ckpt_path="./checkpoints/best.ckpt", save_dir="visualization"):
    """
    1. Loads the best model.
    2. Selects 3 random slices from the 33-66th percentile.
    3. Plots and saves Original LF, Predicted UNet3D, Ground Truth HF for each slice.
    """
    os.makedirs(save_dir, exist_ok=True)
    sample_name = Path(lf_path).name.replace('_lowfield.nii', '').replace('.nii', '')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load data and model
    lf_vol, hf_vol = load_pair_resample_normalize(lf_path, hf_path)
    model = load_model(ckpt_path, device=device, base=56)
    
    # Predict the full volume
    print(f"Running prediction for {sample_name}... this may take a moment.")
    pred_vol = predict_volume(model, lf_vol, patch_size=96, stride=32, device=device)
    pred_vol = np.clip(pred_vol, 0.0, 1.0)
    
    z_dim = lf_vol.shape[2]
    slice_idxs = random.sample(range(int(z_dim * 0.33), int(z_dim * 0.66)), 3)
    
    for z in slice_idxs:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'{sample_name} - Model Predictions (Slice {z})', fontsize=14)
        
        # Original LF
        axes[0].imshow(lf_vol[:, :, z], cmap='gray')
        axes[0].set_title(f'Original LF')
        axes[0].axis('off')
        
        # UNet3D Prediction
        axes[1].imshow(pred_vol[:, :, z], cmap='gray')
        axes[1].set_title(f'Predicted UNet3D')
        axes[1].axis('off')
        
        # Ground Truth HF
        axes[2].imshow(hf_vol[:, :, z], cmap='gray')
        axes[2].set_title(f'Ground Truth HF')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save figure before showing
        save_path = os.path.join(save_dir, f"{sample_name}_prediction_slice_{z}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved: {save_path}")
        
        # Display the figure
        plt.show()

def visualize_prediction_differences(lf_path, hf_path, ckpt_path="./checkpoints/best.ckpt", save_dir="visualization"):
    """
    1. Loads the best model.
    2. Selects 3 random slices from the 33-66th percentile.
    3. Plots and saves Difference Map, Predicted UNet3D, Ground Truth HF for each slice.
    """
    os.makedirs(save_dir, exist_ok=True)
    sample_name = Path(lf_path).name.replace('_lowfield.nii', '').replace('.nii', '')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load data and model
    lf_vol, hf_vol = load_pair_resample_normalize(lf_path, hf_path)
    model = load_model(ckpt_path, device=device, base=56)
    
    # Predict the full volume
    print(f"Running prediction for {sample_name} differences... this may take a moment.")
    pred_vol = predict_volume(model, lf_vol, patch_size=96, stride=32, device=device)
    pred_vol = np.clip(pred_vol, 0.0, 1.0)
    
    # Calculate absolute difference
    diff_vol = np.abs(pred_vol - hf_vol)
    
    z_dim = lf_vol.shape[2]
    slice_idxs = random.sample(range(int(z_dim * 0.33), int(z_dim * 0.66)), 3)
    
    for z in slice_idxs:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'{sample_name} - Prediction Difference Map (Slice {z})', fontsize=14)
        
        # Difference Map
        im = axes[0].imshow(diff_vol[:, :, z], cmap='inferno')
        axes[0].set_title(f'Difference Map')
        axes[0].axis('off')
        fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
        
        # UNet3D Prediction
        axes[1].imshow(pred_vol[:, :, z], cmap='gray')
        axes[1].set_title(f'Predicted UNet3D')
        axes[1].axis('off')
        
        # Ground Truth HF
        axes[2].imshow(hf_vol[:, :, z], cmap='gray')
        axes[2].set_title(f'Ground Truth HF')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save figure before showing
        save_path = os.path.join(save_dir, f"{sample_name}_difference_slice_{z}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved: {save_path}")
        
        # Display the figure
        plt.show()

if __name__ == "__main__":
    val_lf_path = "mri_resolution/train/low_field/sample_011_lowfield.nii"
    val_hf_path = "mri_resolution/train/high_field/sample_011_highfield.nii"
    
    # 1. Plot 5 random pairs
    visualize_random_pairs(val_lf_path, val_hf_path)
    
    # 2. Plot 3 samples: Original, Prediction, Ground Truth
    visualize_model_predictions(val_lf_path, val_hf_path, ckpt_path="./checkpoints/best.ckpt")
    
    # 3. Plot 3 samples: Difference Map, Prediction, Ground Truth
    visualize_prediction_differences(val_lf_path, val_hf_path, ckpt_path="./checkpoints/best.ckpt")