from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from nibabel.processing import resample_from_to

from model import UNet2_5D
from preprocessing import  normalize_lf_for_inference, denormalize_to_lf_scale
from train import predict_volume_batched_xy
from mri_resolution.extract_slices import create_submission_df


def load_model(checkpoint_path="best.ckpt", device="cpu", base=56, use_ema=True, stack_size=7):
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stack_size = 7

    checkpoint_path = "checkpoints_metric/best_overall.ckpt"
    model = load_model(
        checkpoint_path,
        device=device,
        base=128,
        use_ema=True,
        stack_size=stack_size,
    )

    hf_template = get_hf_template()
    test_dir = Path("mri_resolution/test/low_field")

    predictions = {}

    for low_path in sorted(test_dir.glob("*.nii")):
        sample_id = low_path.name.replace("_lowfield.nii", "")

        lf_img = nib.load(str(low_path))
        lf_resampled = resample_from_to(lf_img, hf_template, order=1)

        volume = lf_resampled.get_fdata().astype(np.float32)
        volume, (_,_) = normalize_lf_for_inference(volume)

        pred = predict_volume_batched_xy(
            model,
            volume,
            patch_size=96,
            stride=48,
            device=device,
            stack_size=stack_size,
            use_amp=(device == "cuda"),
            microbatch=32,
        )

        np.clip(pred, 0, 1, out=pred)
        predictions[sample_id] = pred

    df = create_submission_df(predictions)
    df.to_csv("submission.csv", index=False)
    print(f"Saved submission.csv with {len(df)} rows.")


if __name__ == "__main__":
    main()
