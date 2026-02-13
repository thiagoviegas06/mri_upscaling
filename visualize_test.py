"""
Visualize predictions on test data from submission.csv
"""
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# Import your models and preprocessing
import sys
sys.path.insert(0, str(Path(__file__).parent / 'metric_model'))
from model import UNet3D, RefinerUNet3D
from train import predict_volume
import preprocessing


def load_checkpoint(ckpt_path, device='cuda'):
    """Load checkpoint and determine if it has refiner"""
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Create stage1 model
    stage1 = UNet3D(in_ch=1, out_ch=1, base=56).to(device)
    
    # Check if refiner exists
    has_refiner = 'refiner' in ckpt
    refiner = None
    
    if has_refiner:
        print("Checkpoint has refiner (two-stage model)")
        refiner = RefinerUNet3D(in_ch=2, out_ch=1, base=24).to(device)
        refiner.load_state_dict(ckpt['refiner'])
        refiner.eval()
    else:
        print("Checkpoint has stage1 only")
    
    # Load model weights (use EMA if available)
    if 'ema' in ckpt:
        print("Using EMA weights")
        stage1.load_state_dict(ckpt['ema'])
    else:
        stage1.load_state_dict(ckpt['model'])
    
    stage1.eval()
    return stage1, refiner


def load_test_volume(sample_id):
    """Load and preprocess test low-field volume"""
    test_path = Path('mri_resolution/test/low_field')
    lf_path = test_path / f'sample_{sample_id:03d}_lowfield.nii'
    
    if not lf_path.exists():
        raise FileNotFoundError(f"Test file not found: {lf_path}")
    
    # Load and normalize
    lf_vol = nib.load(str(lf_path)).get_fdata().astype(np.float32)
    
    # Apply same normalization as training
    lf_min = lf_vol.min()
    lf_max = lf_vol.max()
    if lf_max > lf_min:
        lf_vol = (lf_vol - lf_min) / (lf_max - lf_min)
    
    return lf_vol


def predict_test_volume(stage1, lf_vol, refiner=None, device='cuda', patch_size=64, overlap=16):
    """Generate prediction on test volume"""
    with torch.no_grad():
        # Stage1 prediction
        pred_stage1 = predict_volume(stage1, lf_vol, device=device, 
                                     patch_size=patch_size, overlap=overlap)
        
        # Refiner if available
        if refiner is not None:
            pred_final = predict_volume(stage1, lf_vol, refiner=refiner, 
                                       device=device, patch_size=patch_size, 
                                       overlap=overlap)
            return pred_stage1, pred_final
        else:
            return pred_stage1, None


def visualize_test_predictions(sample_id, stage1, refiner=None, device='cuda', 
                               slice_axis=2, num_slices=5):
    """
    Visualize predictions on a test sample
    
    Args:
        sample_id: Test sample ID (019, 020, 021, 022, or 023)
        stage1: Stage1 model
        refiner: Optional refiner model
        device: torch device
        slice_axis: Which axis to slice (0, 1, or 2)
        num_slices: Number of slices to show
    """
    print(f"\nProcessing sample_{sample_id:03d}...")
    
    # Load test data
    lf_vol = load_test_volume(sample_id)
    print(f"Loaded low-field volume: {lf_vol.shape}")
    
    # Generate predictions
    print("Generating predictions...")
    pred_stage1, pred_final = predict_test_volume(stage1, lf_vol, refiner, device)
    
    # Select slices to visualize
    total_slices = lf_vol.shape[slice_axis]
    slice_indices = np.linspace(total_slices // 4, 3 * total_slices // 4, 
                               num_slices, dtype=int)
    
    # Setup figure
    has_refiner = pred_final is not None
    n_cols = 3 if has_refiner else 2
    fig, axes = plt.subplots(num_slices, n_cols, figsize=(4 * n_cols, 3 * num_slices))
    
    if num_slices == 1:
        axes = axes.reshape(1, -1)
    
    for i, slice_idx in enumerate(slice_indices):
        # Extract slices
        if slice_axis == 0:
            lf_slice = lf_vol[slice_idx, :, :]
            s1_slice = pred_stage1[slice_idx, :, :]
            rf_slice = pred_final[slice_idx, :, :] if has_refiner else None
        elif slice_axis == 1:
            lf_slice = lf_vol[:, slice_idx, :]
            s1_slice = pred_stage1[:, slice_idx, :]
            rf_slice = pred_final[:, slice_idx, :] if has_refiner else None
        else:  # axis 2
            lf_slice = lf_vol[:, :, slice_idx]
            s1_slice = pred_stage1[:, :, slice_idx]
            rf_slice = pred_final[:, :, slice_idx] if has_refiner else None
        
        # Plot
        axes[i, 0].imshow(lf_slice.T, cmap='gray', origin='lower')
        axes[i, 0].set_title(f'Low-Field (slice {slice_idx})')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(s1_slice.T, cmap='gray', origin='lower')
        axes[i, 1].set_title('Stage1 Prediction')
        axes[i, 1].axis('off')
        
        if has_refiner:
            axes[i, 2].imshow(rf_slice.T, cmap='gray', origin='lower')
            axes[i, 2].set_title('Refiner Prediction')
            axes[i, 2].axis('off')
    
    plt.suptitle(f'Test Predictions - sample_{sample_id:03d}', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Configuration
    CHECKPOINT_PATH = 'metric_model/best_overall.ckpt'  # or 'best.ckpt' or 'best_refiner.ckpt'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test sample IDs from submission.csv
    TEST_SAMPLES = [19, 20, 21, 22, 23]
    
    # Load model
    print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
    stage1, refiner = load_checkpoint(CHECKPOINT_PATH, DEVICE)
    
    # Visualize predictions for each test sample
    # (you can comment out samples if you only want to see one or two)
    for sample_id in TEST_SAMPLES:
        try:
            visualize_test_predictions(
                sample_id=sample_id,
                stage1=stage1,
                refiner=refiner,
                device=DEVICE,
                slice_axis=2,  # axial slices
                num_slices=3   # show 3 slices per sample
            )
        except FileNotFoundError as e:
            print(f"Skipping sample {sample_id}: {e}")
