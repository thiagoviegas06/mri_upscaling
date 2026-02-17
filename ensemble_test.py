from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from nibabel.processing import resample_from_to

from model import UNet2_5D
from preprocessing import preprocess_volume
from train import predict_volume
from mri_resolution.extract_slices import create_submission_df


def load_model(checkpoint_path, device="cpu", base=56, use_ema=True, stack_size=7):
    model = UNet2_5D(in_ch=stack_size, base=base).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and use_ema and "ema" in ckpt:
        state = ckpt["ema"]
    else:
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model




def get_hf_template(hf_dir="mri_resolution/train/high_field"):
    hf_dir = Path(hf_dir)
    hf_files = sorted(list(hf_dir.glob("*.nii")))
    if not hf_files:
        raise FileNotFoundError(f"No HF template found in {hf_dir}")
    return nib.load(str(hf_files[0]))


def ensemble_predict(models, volume, patch_size=96, stride=32, device="cpu", stack_size=7):
    """Predict and average across multiple models."""
    predictions = []
    for model in models:
        pred = predict_volume(model, volume, patch_size=patch_size, stride=stride, device=device, stack_size=stack_size)
        predictions.append(pred)
    
    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stack_size = 7  # Match the stack_size used in main.py
    
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
            model = load_model(ckpt_path, device=device, base=56, stack_size=stack_size)
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

        pred = ensemble_predict(models, volume, patch_size=96, stride=96 // 3, device=device, stack_size=stack_size)
        pred = np.clip(pred, 0.0, 1.0)
        predictions[sample_id] = pred

    df = create_submission_df(predictions)
    df.to_csv("submission_ensemble.csv", index=False)
    print(f"Saved submission_ensemble.csv with {len(df)} rows.")


if __name__ == "__main__":
    main()
