from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from nibabel.processing import resample_from_to

from model import UNet3D
from preprocessing import preprocess_volume
from mri_resolution.extract_slices import create_submission_df


def load_model(checkpoint_path, device="cpu", base=56, use_ema=True):
    model = UNet3D(base=base).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and use_ema and "ema" in ckpt:
        state = ckpt["ema"]
    else:
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


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


def get_hf_template(hf_dir="mri_resolution/train/high_field"):
    hf_dir = Path(hf_dir)
    hf_files = sorted(list(hf_dir.glob("*.nii")))
    if not hf_files:
        raise FileNotFoundError(f"No HF template found in {hf_dir}")
    return nib.load(str(hf_files[0]))


def ensemble_predict(models, volume, patch_size=96, stride=32, device="cpu"):
    """Predict and average across multiple models."""
    predictions = []
    for model in models:
        pred = predict_volume(model, volume, patch_size=patch_size, stride=stride, device=device)
        predictions.append(pred)
    
    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # List of checkpoint paths to ensemble
    checkpoint_paths = [
        "checkpoints/best.ckpt",
        "checkpoints/epoch_25.ckpt",
        "checkpoints/epoch_20.ckpt",
    ]
    
    # Load all models
    print("Loading models...")
    models = []
    for ckpt_path in checkpoint_paths:
        if Path(ckpt_path).exists():
            model = load_model(ckpt_path, device=device, base=56)
            models.append(model)
            print(f"  Loaded {ckpt_path}")
        else:
            print(f"  Warning: {ckpt_path} not found, skipping")
    
    if not models:
        raise RuntimeError("No valid checkpoints found for ensemble")
    
    print(f"Ensemble size: {len(models)} models")
    
    hf_template = get_hf_template()
    test_dir = Path("mri_resolution/test/low_field")
    predictions = {}

    for low_path in sorted(test_dir.glob("*.nii")):
        sample_id = low_path.name.replace("_lowfield.nii", "")
        print(f"Processing {sample_id}...")
        
        lf_img = nib.load(str(low_path))
        lf_resampled = resample_from_to(lf_img, hf_template, order=1)
        volume = lf_resampled.get_fdata().astype(np.float32)
        volume = preprocess_volume(volume)

        pred = ensemble_predict(models, volume, patch_size=96, stride=96 // 3, device=device)
        pred = np.clip(pred, 0.0, 1.0)
        predictions[sample_id] = pred

    df = create_submission_df(predictions)
    df.to_csv("submission_ensemble.csv", index=False)
    print(f"Saved submission_ensemble.csv with {len(df)} rows.")


if __name__ == "__main__":
    main()
