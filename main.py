from mri_resolution.extract_slices import (
    load_nifti,
    slice_to_base64,
    base64_to_slice,
    volume_to_submission_rows,
    create_submission_df
)

import os, random
import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from model import UNet3D
from train import train_one_epoch, validate, validate_full_volume
from preprocessing import MRIPatchDataset

def make_pairs(lf_dir, hf_dir):
    # Build paired (LF, HF) file list based on naming convention.
    pairs = []
    for fname in sorted(os.listdir(lf_dir)):
        if not (fname.endswith(".nii") or fname.endswith(".nii.gz")):
            continue
        lf_path = os.path.join(lf_dir, fname)
        hf_name = fname.replace("lowfield", "highfield")
        hf_path = os.path.join(hf_dir, hf_name)
        if os.path.exists(hf_path):
            pairs.append((lf_path, hf_path))
    return pairs

def split_pairs(pairs, val_frac=0.2, seed=42):
    # Shuffle and split pairs into train/val subsets.
    pairs = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_frac))
    return pairs[n_val:], pairs[:n_val]

if __name__ == "__main__":
    patch_size = 96

    pairs = make_pairs("mri_resolution/train/low_field", "mri_resolution/train/high_field")
    train_pairs, val_pairs = split_pairs(pairs, val_frac=0.2, seed=42)

    print("Num pairs:", len(pairs))
    print("Train pairs:", len(train_pairs), "Val pairs:", len(val_pairs))
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_ds = MRIPatchDataset(train_pairs, patch_size=patch_size, patches_per_volume=64, cache_volumes=True)
    val_ds   = MRIPatchDataset(val_pairs,   patch_size=patch_size, patches_per_volume=16, cache_volumes=True)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=2, shuffle=False, num_workers=2, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet3D(base=48).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = GradScaler("cuda") if device == "cuda" else None

    best_val = float("inf")
    save_dir = os.environ.get("MODEL_DESTINATION", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "best.ckpt")

    pretrain_epochs = 50
    finetune_epochs = 20

    pretrain_weights = {"l1": 1.0, "l2": 0.0, "ssim": 1.0}
    finetune_weights = {"l1": 0.3, "l2": 0.5, "ssim": 0.2}
    
    val_full_every = 5
    val_full_max_volumes = 2

    for epoch in range(1, pretrain_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optim, device, scaler, loss_weights=pretrain_weights)
        val_metrics   = validate(model, val_loader, device, loss_weights=pretrain_weights)

        if not torch.isfinite(torch.tensor(train_metrics["loss"])) or train_metrics["loss"] > 5.0:
            print("Stopping: training loss exploded.")
            break

        msg = (
            f"pretrain {epoch:02d} | loss: {train_metrics['loss']:.5f} "
            f"| l1: {train_metrics['l1']:.5f} | l2: {train_metrics['l2']:.5f} "
            f"| ssim: {train_metrics['ssim']:.5f} | val: {val_metrics['loss']:.5f} "
            f"| val_l1: {val_metrics['l1']:.5f} | val_l2: {val_metrics['l2']:.5f} "
            f"| val_ssim: {val_metrics['ssim']:.5f}"
        )
        if epoch % val_full_every == 0:
            val_full = validate_full_volume(
                model,
                val_pairs,
                device,
                loss_weights=pretrain_weights,
                patch_size=patch_size,
                stride=patch_size // 2,
                max_volumes=val_full_max_volumes,
            )
            msg += (
                f" | val_full: {val_full['loss']:.5f}"
                f" | val_full_l1: {val_full['l1']:.5f}"
                f" | val_full_l2: {val_full['l2']:.5f}"
                f" | val_full_ssim: {val_full['ssim']:.5f}"
            )
        print(msg)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "optim": optim.state_dict(), "val_loss": val_metrics["loss"]},
                best_path
            )
            print("Saved best to:", best_path)

    for epoch in range(1, finetune_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optim, device, scaler, loss_weights=finetune_weights)
        val_metrics   = validate(model, val_loader, device, loss_weights=finetune_weights)

        if not torch.isfinite(torch.tensor(train_metrics["loss"])) or train_metrics["loss"] > 5.0:
            print("Stopping: training loss exploded.")
            break

        msg = (
            f"finetune {epoch:02d} | loss: {train_metrics['loss']:.5f} "
            f"| l1: {train_metrics['l1']:.5f} | l2: {train_metrics['l2']:.5f} "
            f"| ssim: {train_metrics['ssim']:.5f} | val: {val_metrics['loss']:.5f} "
            f"| val_l1: {val_metrics['l1']:.5f} | val_l2: {val_metrics['l2']:.5f} "
            f"| val_ssim: {val_metrics['ssim']:.5f}"
        )
        if epoch % val_full_every == 0:
            val_full = validate_full_volume(
                model,
                val_pairs,
                device,
                loss_weights=finetune_weights,
                patch_size=patch_size,
                stride=patch_size // 2,
                max_volumes=val_full_max_volumes,
            )
            msg += (
                f" | val_full: {val_full['loss']:.5f}"
                f" | val_full_l1: {val_full['l1']:.5f}"
                f" | val_full_l2: {val_full['l2']:.5f}"
                f" | val_full_ssim: {val_full['ssim']:.5f}"
            )
        print(msg)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {"epoch": pretrain_epochs + epoch, "model": model.state_dict(), "optim": optim.state_dict(), "val_loss": val_metrics["loss"]},
                best_path
            )
            print("Saved best to:", best_path)