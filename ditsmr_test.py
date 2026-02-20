from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from nibabel.processing import resample_from_to
from tqdm import tqdm
import pandas as pd

# Import the new DiTMSR model
from model import DiTMSR
from preprocessing import preprocess_volume

# Keep the submission helper if available, otherwise define a placeholder
try:
    from mri_resolution.extract_slices import create_submission_df
except ImportError:
    def create_submission_df(predictions):
        # Fallback implementation based on standard submission formats
        print("Warning: mri_resolution.extract_slices not found. Returning simple DataFrame.")
        return pd.DataFrame(list(predictions.items()), columns=['id', 'prediction'])

def load_ditmsr_model(ckpt_path, device="cpu"):
    """
    Loads the DiTMSR model.
    """
    model = DiTMSR(device=device)
    if ckpt_path and Path(ckpt_path).exists():
        print(f"Loading checkpoint from {ckpt_path}...")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Checkpoint {ckpt_path} not found. Using random initialization.")
    
    model.to(device)
    model.eval()
    return model

# --- Sliding Window Helpers ---

def _start_indices(dim, patch_size, stride):
    if dim <= patch_size:
        return [0]
    idxs = list(range(0, dim - patch_size + 1, stride))
    if idxs[-1] != dim - patch_size:
        idxs.append(dim - patch_size)
    return idxs

def _gaussian_window_3d(patch_size, sigma=None):
    if sigma is None:
        sigma = patch_size / 5.0
    coords = np.arange(patch_size) - (patch_size - 1) / 2.0
    g1d = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    g3d = g1d[:, None, None] * g1d[None, :, None] * g1d[None, None, :]
    return g3d.astype(np.float32)

def predict_volume(model, volume, patch_size=64, stride=32, device="cpu", steps=15):
    """
    Sliding window inference for DiTMSR.
    DiTMSR is a 2D model. We treat 3D patches as batches of 2D slices.
    Volume input shape: (X, Y, Z)
    """
    # The model is hardcoded for 64x64 inputs in model.py (latent 16x16, patch 2 -> 64/4=16)
    # So we force patch_size to 64 for spatial dims.
    # We can handle arbitrary depth (Z) in the batch dimension.
    
    spatial_patch_size = 64 # Constraint of the DiT architecture in model.py
    
    # Check dimensions
    if volume.shape[0] < spatial_patch_size or volume.shape[1] < spatial_patch_size:
        # Pad if volume is smaller than patch size
        pad_x = max(0, spatial_patch_size - volume.shape[0])
        pad_y = max(0, spatial_patch_size - volume.shape[1])
        volume = np.pad(volume, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

    x_starts = _start_indices(volume.shape[0], spatial_patch_size, stride)
    y_starts = _start_indices(volume.shape[1], spatial_patch_size, stride)
    z_starts = _start_indices(volume.shape[2], spatial_patch_size, stride) # Chunking Z for memory

    accum = np.zeros_like(volume, dtype=np.float32)
    weight = np.zeros_like(volume, dtype=np.float32)
    
    # 3D Gaussian window (used to blend batches in Z and tiles in XY)
    gaussian_window = _gaussian_window_3d(spatial_patch_size)

    with torch.no_grad():
        for x in x_starts:
            for y in y_starts:
                for z in z_starts:
                    # Extract 3D Patch: (64, 64, 64)
                    patch = volume[x:x + spatial_patch_size, y:y + spatial_patch_size, z:z + spatial_patch_size]
                    
                    # Shape: (X, Y, Z) -> Need (Batch=Z, C=1, H=X, W=Y) for 2D model
                    # Note: We process slices along Z axis.
                    patch_t = torch.from_numpy(patch).to(device)
                    # Permute to (Z, 1, X, Y)
                    patch_t = patch_t.permute(2, 0, 1).unsqueeze(1) # (Z, 1, 64, 64)
                    
                    # Inference
                    # In DiTMSR inference, we use the input LR image as both:
                    # 1. The structural guidance (lr_img argument -> decoder)
                    # 2. The conditioning for diffusion (ref_img argument -> DiT condition)
                    # This assumes Single Contrast SR setup where LR is the only available reference.
                    sr_t = model.forward_inference(lr_img=patch_t, ref_img=patch_t, steps=steps)
                    
                    # Output is (Z, 1, 64, 64)
                    # Permute back to (X, Y, Z)
                    # (Z, 1, X, Y) -> squeeze -> (Z, X, Y) -> permute -> (X, Y, Z)
                    pred = sr_t.squeeze(1).permute(1, 2, 0).cpu().numpy()

                    # Accumulate
                    accum[x:x + spatial_patch_size, y:y + spatial_patch_size, z:z + spatial_patch_size] += pred * gaussian_window
                    weight[x:x + spatial_patch_size, y:y + spatial_patch_size, z:z + spatial_patch_size] += gaussian_window

    return accum / np.maximum(weight, 1e-8)

def get_hf_template(hf_dir="mri_resolution/train/high_field"):
    hf_dir = Path(hf_dir)
    # Create directory if it doesn't exist to prevent immediate crash, though user should provide data
    if not hf_dir.exists():
        print(f"Warning: {hf_dir} does not exist. Please ensure data is available.")
        return None
        
    hf_files = sorted(list(hf_dir.glob("*.nii")))
    if not hf_files:
        raise FileNotFoundError(f"No HF template found in {hf_dir}")
    return nib.load(str(hf_files[0]))

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Path to your trained checkpoint
    # Adjust this to point to your actual trained weights
    checkpoint_path = "checkpoints/stage2_epoch_50.pth" 
    
    model = load_ditmsr_model(
        ckpt_path=checkpoint_path, 
        device=device
    )

    hf_template = get_hf_template()
    if hf_template is None:
        print("Skipping inference due to missing HF template.")
        return

    test_dir = Path("mri_resolution/test/low_field")
    if not test_dir.exists():
        print(f"Test directory {test_dir} not found.")
        return

    predictions = {}

    print("Running Inference with DiTMSR Model...")
    # Using a reduced patch size (64) and stride (32) compared to original 96/48
    # because DiTMSR architecture in model.py is hardcoded for 64x64 patches.
    PATCH_SIZE = 64 
    STRIDE = 32
    DIFFUSION_STEPS = 15

    for low_path in tqdm(sorted(test_dir.glob("*.nii"))):
        sample_id = low_path.name.replace("_lowfield.nii", "")
        lf_img = nib.load(str(low_path))
        
        # Resample to match High Field resolution
        lf_resampled = resample_from_to(lf_img, hf_template, order=1)
        volume = lf_resampled.get_fdata().astype(np.float32)
        volume = preprocess_volume(volume)

        # Predict using sliding window
        pred = predict_volume(
            model, 
            volume, 
            patch_size=PATCH_SIZE, 
            stride=STRIDE, 
            device=device,
            steps=DIFFUSION_STEPS
        )
        
        pred = np.clip(pred, 0.0, 1.0)
        predictions[sample_id] = pred

    # Generate submission file
    try:
        df = create_submission_df(predictions)
        df.to_csv("submission.csv", index=False)
        print(f"Saved submission.csv with {len(df)} rows.")
    except Exception as e:
        print(f"Error creating submission dataframe: {e}")
        # Fallback save just in case
        import pickle
        with open("predictions.pkl", "wb") as f:
            pickle.dump(predictions, f)
        print("Saved raw predictions to predictions.pkl instead.")

if __name__ == "__main__":
    main()