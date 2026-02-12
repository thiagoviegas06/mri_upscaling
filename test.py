from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from nibabel.processing import resample_from_to
from tqdm import tqdm

# Import the wrapper that combines UNet + Refiner
from refiner_model import RRDBNet, CascadedModel
from model import UNet3D
from preprocessing import preprocess_volume
from mri_resolution.extract_slices import create_submission_df

def load_cascaded_model(unet_path="checkpoints/best.ckpt", refiner_path="checkpoints/refiner_best.ckpt", device="cpu"):
    """
    Loads both models and wraps them into a single CascadedModel
    that behaves like a standard 3D model (Input: 3D Patch -> Output: 3D Patch).
    """
    # 1. Load UNet
    unet = UNet3D(base=56).to(device)
    unet_ckpt = torch.load(unet_path, map_location=device)
    # Handle potentially different checkpoint structures
    unet_state = unet_ckpt["ema"] if "ema" in unet_ckpt else unet_ckpt["model"]
    unet.load_state_dict(unet_state)
    
    # 2. Load Refiner
    refiner = RRDBNet(in_nc=1, out_nc=1, nf=64, nb=4).to(device)
    refiner_ckpt = torch.load(refiner_path, map_location=device)
    refiner_state = refiner_ckpt["ema"] if "ema" in refiner_ckpt else (refiner_ckpt["model"] if "model" in refiner_ckpt else refiner_ckpt)
    refiner.load_state_dict(refiner_state)
    
    # 3. Wrap
    model = CascadedModel(unet, refiner, device=device)
    model.eval()
    return model

# --- Sliding Window Helpers (Restored) ---

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

def predict_volume(model, volume, patch_size=96, stride=48, device="cpu"):
    """
    Standard sliding window inference. 
    Because 'model' is now the CascadedModel, passing a 3D patch here 
    will automatically trigger: 3D UNet -> Slice Split -> 2D Refiner -> 3D Merge.
    """
    x_starts = _start_indices(volume.shape[0], patch_size, stride)
    y_starts = _start_indices(volume.shape[1], patch_size, stride)
    z_starts = _start_indices(volume.shape[2], patch_size, stride)

    accum = np.zeros_like(volume, dtype=np.float32)
    weight = np.zeros_like(volume, dtype=np.float32)
    gaussian_window = _gaussian_window_3d(patch_size)

    with torch.no_grad():
        for x in x_starts:
            for y in y_starts:
                for z in z_starts:
                    # Extract 3D Patch
                    patch = volume[x:x + patch_size, y:y + patch_size, z:z + patch_size]
                    
                    # Convert to tensor (B, C, D, H, W)
                    patch_t = torch.from_numpy(patch)[None, None, ...].to(device)
                    
                    # Run Cascade (UNet + Refiner)
                    pred_t = model(patch_t)
                    
                    # Back to numpy
                    pred = pred_t.squeeze(0).squeeze(0).cpu().numpy()

                    # Accumulate
                    accum[x:x + patch_size, y:y + patch_size, z:z + patch_size] += pred * gaussian_window
                    weight[x:x + patch_size, y:y + patch_size, z:z + patch_size] += gaussian_window

    return accum / np.maximum(weight, 1e-8)

def get_hf_template(hf_dir="mri_resolution/train/high_field"):
    hf_dir = Path(hf_dir)
    hf_files = sorted(list(hf_dir.glob("*.nii")))
    if not hf_files:
        raise FileNotFoundError(f"No HF template found in {hf_dir}")
    return nib.load(str(hf_files[0]))

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the combined pipeline
    # Ensure paths match where your train_refiner.py saved them
    model = load_cascaded_model(
        unet_path="checkpoints/best.ckpt", 
        refiner_path="checkpoints/refiner_best.ckpt", 
        device=device
    )

    hf_template = get_hf_template()
    test_dir = Path("mri_resolution/test/low_field")
    predictions = {}

    print("Running Inference with Cascaded Model...")
    for low_path in tqdm(sorted(test_dir.glob("*.nii"))):
        sample_id = low_path.name.replace("_lowfield.nii", "")
        lf_img = nib.load(str(low_path))
        
        # Resample to match High Field resolution
        lf_resampled = resample_from_to(lf_img, hf_template, order=1)
        volume = lf_resampled.get_fdata().astype(np.float32)
        volume = preprocess_volume(volume)

        # Predict using sliding window on the CascadedModel
        pred = predict_volume(model, volume, patch_size=96, stride=48, device=device)
        pred = np.clip(pred, 0.0, 1.0)
        predictions[sample_id] = pred

    df = create_submission_df(predictions)
    df.to_csv("submission.csv", index=False)
    print(f"Saved submission.csv with {len(df)} rows.")

if __name__ == "__main__":
    main()