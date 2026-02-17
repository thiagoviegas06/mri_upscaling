from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import torch
from nibabel.processing import resample_from_to

from model import UNet2_5D
from preprocessing import preprocess_volume
from train import predict_volume
from mri_resolution.extract_slices import volume_to_submission_rows
from mri_resolution import metric as eval_metric


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


def build_submission_df(model, pairs, device, patch_size=96, stride=48, stack_size=7):
    rows = []
    for lf_path, hf_path in pairs:
        sample_id = Path(lf_path).name.replace("_lowfield.nii", "")
        hf_img = nib.load(str(hf_path))
        lf_img = nib.load(str(lf_path))
        lf_resampled = resample_from_to(lf_img, hf_img, order=1)
        volume = lf_resampled.get_fdata().astype(np.float32)
        volume = preprocess_volume(volume)

        pred = predict_volume(model, volume, patch_size=patch_size, stride=stride, device=device, stack_size=stack_size)
        pred = np.clip(pred, 0.0, 1.0)
        rows.extend(volume_to_submission_rows(pred, sample_id))
    return pd.DataFrame(rows)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stack_size = 7  # Match the stack_size used in main.py
    
    # List of checkpoint paths to evaluate
    checkpoint_paths = [
        "checkpoints_metric/best.ckpt",
        "checkpoints_metric/best_overall.ckpt",
        "checkpoints_metric/epoch_05.ckpt",
        "checkpoints_metric/epoch_10.ckpt",
        "checkpoints_metric/epoch_15.ckpt",
        "checkpoints_metric/epoch_20.ckpt",
        "checkpoints_metric/epoch_25.ckpt",
        "checkpoints_metric/epoch_30.ckpt",
        "checkpoints_metric/epoch_35.ckpt",
        "checkpoints_metric/epoch_40.ckpt",
    ]
    
    pairs = make_pairs("mri_resolution/train/low_field", "mri_resolution/train/high_field")
    if not pairs:
        raise RuntimeError("No LF/HF pairs found for validation.")

    solution = build_solution_df(pairs)
    
    print("Evaluating individual models...\n")
    scores = []
    
    for ckpt_path in checkpoint_paths:
        if not Path(ckpt_path).exists():
            print(f"  Skipping {ckpt_path} (not found)")
            continue
        
        print(f"  Loading {ckpt_path}...")
        model = load_model(ckpt_path, device=device, base=56, stack_size=stack_size)
        
        print(f"  Evaluating {ckpt_path}...")
        submission = build_submission_df(model, pairs, device=device, patch_size=96, stride=96 // 3, stack_size=stack_size)
        
        score = eval_metric.score(solution, submission, "row_id")
        scores.append((ckpt_path, score))
        print(f"  MS-SSIM: {score:.5f}\n")
    
    print("\n" + "="*60)
    print("Summary of model scores (MS-SSIM):")
    print("="*60)
    for ckpt_path, score in scores:
        model_name = Path(ckpt_path).name
        print(f"  {model_name:30s} : {score:.5f}")
    
    if scores:
        best_path, best_score = max(scores, key=lambda x: x[1])
        print(f"\nBest model: {Path(best_path).name} with MS-SSIM: {best_score:.5f}")


if __name__ == "__main__":
    main()
