import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import nibabel as nib
from nibabel.processing import resample_from_to

# Import from your existing modules
from model import UNet3D
from preprocessing import load_pair_resample_normalize
from test import predict_volume_unet

def load_model(checkpoint_path, device="cpu", base=16):
    """Loads the UNet3D model from the specified checkpoint."""
    model = UNet3D(base=base).to(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Handle state dict wrapped in "model" key or bare state dict
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model

def make_pairs(lf_dir, hf_dir):
    """Scans directories to create (LF, HF) pairs."""
    pairs = []
    # Using the standard naming convention from your project
    for lf_path in sorted(Path(lf_dir).glob("*.nii")):
        hf_name = lf_path.name.replace("lowfield", "highfield")
        hf_path = Path(hf_dir) / hf_name
        if hf_path.exists():
            pairs.append((lf_path, hf_path))
    return pairs

def visualize_comparison(pred_vol, gt_vol, subject_id, output_dir="visualizations"):
    """
    Visualizes 5 slices of the generated volume vs ground truth.
    Saves the result to output_dir.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Select 5 slices evenly spaced across the Z-axis (avoiding very ends)
    depth = pred_vol.shape[2]
    slice_indices = np.linspace(10, depth - 10, num=5, dtype=int)
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f"Subject: {subject_id} (Top: Ground Truth, Bottom: Prediction)", fontsize=16)

    for i, z in enumerate(slice_indices):
        # Ground Truth Slice
        axes[0, i].imshow(gt_vol[:, :, z], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"Slice {z} (GT)")
        axes[0, i].axis("off")
        
        # Predicted Slice
        axes[1, i].imshow(pred_vol[:, :, z], cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"Slice {z} (Pred)")
        axes[1, i].axis("off")

    save_path = os.path.join(output_dir, f"{subject_id}_comparison.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved visualization to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Run inference and visualize results for MRI Upscaling.")
    parser.add_argument("model_path", type=str, help="Path to the trained model checkpoint (e.g., best.ckpt)")
    parser.add_argument("--data_dir", type=str, default="mri_resolution/train", help="Base directory containing low_field and high_field folders")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Directory to save output images")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    
    args = parser.parse_args()

    # Define paths
    lf_dir = Path(args.data_dir) / "low_field"
    hf_dir = Path(args.data_dir) / "high_field"

    # Load Model
    print(f"Loading model from {args.model_path} on {args.device}...")
    model = load_model(args.model_path, device=args.device)

    # Get Pairs
    pairs = make_pairs(lf_dir, hf_dir)
    if not pairs:
        print(f"No pairs found in {args.data_dir}. Please check your data directory structure.")
        return

    print(f"Found {len(pairs)} subjects. Starting inference...")

    # Inference Loop
    for i, (lf_path, hf_path) in enumerate(pairs):
        subject_id = lf_path.name.replace("_lowfield.nii", "")
        print(f"Processing subject {i+1}/{len(pairs)}: {subject_id}")

        # Load and Preprocess
        # load_pair_resample_normalize handles the resampling of LF to HF dimensions
        lf, hf = load_pair_resample_normalize(str(lf_path), str(hf_path))

        # Predict
        # Using predict_volume from test.py/train.py logic
        pred = predict_volume_unet(model, lf, patch_size=96, stride=32, device=args.device)
        pred = np.clip(pred, 0.0, 1.0)

        # Visualize
        visualize_comparison(pred, hf, subject_id, output_dir=args.output_dir)

if __name__ == "__main__":
    main()