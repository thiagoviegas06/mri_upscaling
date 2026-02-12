from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from nibabel.processing import resample_from_to
from tqdm import tqdm

from model import UNet3D
from refiner_model import RRDBNet
from preprocessing import preprocess_volume
from mri_resolution.extract_slices import create_submission_df

def load_models(unet_path="checkpoints/best.ckpt", refiner_path="checkpoints/refiner_best.ckpt", device="cpu"):
    # Load UNet
    unet = UNet3D(base=56).to(device)
    unet_ckpt = torch.load(unet_path, map_location=device)
    unet_state = unet_ckpt["ema"] if "ema" in unet_ckpt else unet_ckpt["model"]
    unet.load_state_dict(unet_state)
    unet.eval()
    
    # Load Refiner
    refiner = RRDBNet(in_nc=1, out_nc=1, nf=64, nb=4).to(device) # Ensure nb matches training
    if Path(refiner_path).exists():
        refiner_ckpt = torch.load(refiner_path, map_location=device)
        refiner.load_state_dict(refiner_ckpt["model"])
    else:
        print("Warning: Refiner checkpoint not found, skipping refinement weights.")
    refiner.eval()
    
    return unet, refiner

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

def predict_volume_unet(model, volume, patch_size=96, stride=48, device="cpu"):
    """Stage 1: 3D UNet Prediction"""
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
                    patch = volume[x:x + patch_size, y:y + patch_size, z:z + patch_size]
                    patch_t = torch.from_numpy(patch)[None, None, ...].to(device)
                    pred_t = model(patch_t)
                    pred = pred_t.squeeze(0).squeeze(0).cpu().numpy()

                    accum[x:x + patch_size, y:y + patch_size, z:z + patch_size] += pred * gaussian_window
                    weight[x:x + patch_size, y:y + patch_size, z:z + patch_size] += gaussian_window

    return accum / np.maximum(weight, 1e-8)

def refine_volume_slices(refiner, volume, batch_size=16, device="cpu"):
    """Stage 2: 2D Slice Refinement (Axial/Z-axis)"""
    D, H, W = volume.shape
    refined_volume = np.zeros_like(volume)
    
    with torch.no_grad():
        # Process in batches of slices
        for i in range(0, D, batch_size):
            end = min(i + batch_size, D)
            slices = volume[i:end, :, :] # (B, H, W)
            
            # Prepare tensor: (B, 1, H, W)
            slices_t = torch.from_numpy(slices).unsqueeze(1).to(device)
            
            # Refine
            refined_t = refiner(slices_t)
            refined_batch = refined_t.squeeze(1).cpu().numpy()
            
            refined_volume[i:end, :, :] = refined_batch
            
    return refined_volume

def get_hf_template(hf_dir="mri_resolution/train/high_field"):
    hf_dir = Path(hf_dir)
    hf_files = sorted(list(hf_dir.glob("*.nii")))
    if not hf_files:
        raise FileNotFoundError(f"No HF template found in {hf_dir}")
    return nib.load(str(hf_files[0]))

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet, refiner = load_models(device=device)

    hf_template = get_hf_template()

    test_dir = Path("mri_resolution/test/low_field")
    predictions = {}

    print("Starting Inference Pipeline: UNet3D -> RRDBRefiner...")
    for low_path in tqdm(sorted(test_dir.glob("*.nii"))):
        sample_id = low_path.name.replace("_lowfield.nii", "")
        lf_img = nib.load(str(low_path))
        lf_resampled = resample_from_to(lf_img, hf_template, order=1)
        volume = lf_resampled.get_fdata().astype(np.float32)
        volume = preprocess_volume(volume)

        # Stage 1: UNet
        unet_pred = predict_volume_unet(unet, volume, patch_size=96, stride=32, device=device)
        
        # Stage 2: Refiner
        # We iterate over the Depth (Z) axis. If you want X or Y, change indexing.
        refined_pred = refine_volume_slices(refiner, unet_pred, device=device)

        final_pred = np.clip(refined_pred, 0.0, 1.0)
        predictions[sample_id] = final_pred

    df = create_submission_df(predictions)
    df.to_csv("submission.csv", index=False)
    print(f"Saved submission.csv with {len(df)} rows.")

if __name__ == "__main__":
    main()