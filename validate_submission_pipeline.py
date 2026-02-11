from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import torch
from nibabel.processing import resample_from_to

from model import UNet3D
from preprocessing import preprocess_volume
from mri_resolution.extract_slices import volume_to_submission_rows
from mri_resolution import metric as eval_metric
from test import predict_volume


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


def make_pairs(lf_dir, hf_dir):
    pairs = []
    for lf_path in sorted(Path(lf_dir).glob("*.nii")):
        hf_name = lf_path.name.replace("lowfield", "highfield")
        hf_path = Path(hf_dir) / hf_name
        if hf_path.exists():
            pairs.append((lf_path, hf_path))
    return pairs


def build_solution_df(pairs):
    rows = []
    for lf_path, hf_path in pairs:
        sample_id = lf_path.name.replace("_lowfield.nii", "")
        hf_img = nib.load(str(hf_path))
        hf_vol = hf_img.get_fdata().astype(np.float32)
        rows.extend(volume_to_submission_rows(hf_vol, sample_id))
    df = pd.DataFrame(rows)
    df = df.rename(columns={"prediction": "ground_truth"})
    return df


def ensemble_predict(models, volume, patch_size=96, stride=48, device="cpu"):
    """Predict and average across multiple models."""
    predictions = []
    for model in models:
        pred = predict_volume(model, volume, patch_size=patch_size, stride=stride, device=device)
        predictions.append(pred)
    
    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred


def build_submission_df(model, pairs, device, patch_size=96, stride=48):
    rows = []
    for lf_path, hf_path in pairs:
        sample_id = Path(lf_path).name.replace("_lowfield.nii", "")
        hf_img = nib.load(str(hf_path))
        lf_img = nib.load(str(lf_path))
    
    # Set to True to use ensemble, False for single model
    use_ensemble = True
    
    if use_ensemble:
        # List of checkpoint paths to ensemble
        checkpoint_paths = [
            "checkpoints/best.ckpt",
            "checkpoints/epoch_25.ckpt",
            "checkpoints/epoch_20.ckpt",
        ]
        
        # Load all models
        print("Loading models for ensemble...")
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
        
        print(f"Ensemble size: {len(models)} models\n")
    else:
        checkpoint = "checkpoints_metric/best.ckpt"
        model = load_model(checkpoint, device=device, base=56)
        print(f"Using single model: {checkpoint}\n")

    pairs = make_pairs("mri_resolution/train/low_field", "mri_resolution/train/high_field")
    if not pairs:
        raise RuntimeError("No LF/HF pairs found for validation.")

    solution = build_solution_df(pairs)
    
    if use_ensemble:
        submission = build_submission_df_ensemble(models, pairs, device=device, patch_size=96, stride=96 // 3)
    else:
        submission = build_submission_df(model, pairs, device=device, patch_size=96, stride=96 // 3)

    score = eval_metric.score(solution, submission, "row_id")
    print(f"\nValidation score (submission pipeline): {score:.5f}")
    if use_ensemble:
        print(f"Ensemble of {len(models)} models
    for lf_path, hf_path in pairs:
        sample_id = Path(lf_path).name.replace("_lowfield.nii", "")
        print(f"  Processing {sample_id}...")
        hf_img = nib.load(str(hf_path))
        lf_img = nib.load(str(lf_path))
        lf_resampled = resample_from_to(lf_img, hf_img, order=1)
        volume = lf_resampled.get_fdata().astype(np.float32)
        volume = preprocess_volume(volume)

        pred = ensemble_predict(models, volume, patch_size=patch_size, stride=stride, device=device)
        pred = np.clip(pred, 0.0, 1.0)
        rows.extend(volume_to_submission_rows(pred, sample_id))
    return pd.DataFrame(rows)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = "checkpoints_metric/best.ckpt"

    pairs = make_pairs("mri_resolution/train/low_field", "mri_resolution/train/high_field")
    if not pairs:
        raise RuntimeError("No LF/HF pairs found for validation.")

    model = load_model(checkpoint, device=device, base=56)
    solution = build_solution_df(pairs)
    submission = build_submission_df(model, pairs, device=device, patch_size=96, stride=96 // 3)

    score = eval_metric.score(solution, submission, "row_id")
    print(f"Validation score (submission pipeline): {score:.5f}")


if __name__ == "__main__":
    main()
