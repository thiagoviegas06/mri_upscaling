import os
import random
import sys
from pathlib import Path

import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import UNet2_5D
from train import EMA, train_one_epoch, validate, validate_metric
from preprocessing import MRIPatchDataset

def make_pairs(lf_dir, hf_dir):
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
    pairs = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_frac))
    return pairs[n_val:], pairs[:n_val]

if __name__ == "__main__":
    patch_size = 96
    stack_size = 7
    warmup_epochs = 20
    # Since competition metric is MS-SSIM, weight it heavily
    ms_ssim_weight_start = 0.7  # weight for (1 - MS-SSIM) in loss
    l1_weight_start = 0.3      # weight for L1 in loss (helps with convergence)
    ms_ssim_weight_final = 0.85  # Final: mostly MS-SSIM since that's the metric
    l1_weight_final = 0.15

    pairs = make_pairs("/scratch/tjv235/pytorch-example/mri_upscaling/mri_resolution/train/low_field", 
    "/scratch/tjv235/pytorch-example/mri_upscaling/mri_resolution/train/high_field")
    
    train_pairs, val_pairs = split_pairs(pairs, val_frac=0.2, seed=42)

    print("Num pairs:", len(pairs))
    print("Train pairs:", len(train_pairs), "Val pairs:", len(val_pairs))
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_ds = MRIPatchDataset(
        train_pairs,
        patch_size=patch_size,
        patches_per_volume=64,
        cache_volumes=True,
        stack_size=stack_size,
        sample_strategy="filtered",
    )
    val_ds   = MRIPatchDataset(
        val_pairs,
        patch_size=patch_size,
        patches_per_volume=16,
        cache_volumes=True,
        stack_size=stack_size,
    )

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=2, shuffle=False, num_workers=0, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stage1 = UNet2_5D(in_ch=stack_size, base=56).to(device)

    optim1 = torch.optim.AdamW(stage1.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim1,
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=1e-5,
    )
    scaler = GradScaler("cuda") if device == "cuda" else None
    ema = EMA(stage1, decay=0.999)

    best_val = float("-inf")
    best_epoch = 0
    patience = 12
    epochs_no_improve = 0
    save_dir = os.environ.get("MODEL_DESTINATION_METRIC", "checkpoints_metric")
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "best.ckpt")

    num_epochs = 40
    for epoch in range(1, num_epochs + 1):
        # --- inside epoch loop ---
        if warmup_epochs > 0:
            t = min(1.0, (epoch - 1) / max(1, warmup_epochs - 1))  # epoch=1 -> 0.0, epoch=warmup_epochs -> 1.0
        else:
            t = 1.0

        ms_ssim_weight = ms_ssim_weight_start + t * (ms_ssim_weight_final - ms_ssim_weight_start)
        l1_weight      = l1_weight_start      + t * (l1_weight_final      - l1_weight_start)

        train_loss = train_one_epoch(stage1, train_loader, optim1, device, scaler, ema=ema, ms_ssim_weight=ms_ssim_weight, l1_weight=l1_weight)

        ema_backup = ema.apply_to(stage1)
        val_loss = validate(stage1, val_loader, device)  # Optionally update to use ms_ssim_weight if needed
        val_score, val_ms_ssim, val_ssim, val_psnr, val_slices = validate_metric(
            stage1,
            val_pairs,
            device,
            patch_size=patch_size,
            stride=patch_size // 2,
            stack_size=stack_size,
        )
        ema.restore(stage1, ema_backup)

        if epoch % 5 == 0:
            epoch_path = os.path.join(save_dir, f"epoch_{epoch:02d}.ckpt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": stage1.state_dict(),
                    "ema": ema.state_dict(),
                    "optim": optim1.state_dict(),
                    "val_loss": val_loss,
                    "val_score": val_score,
                    "val_ssim": val_ssim,
                    "val_ms_ssim": val_ms_ssim,
                    "val_psnr": val_psnr,
                },
                epoch_path
            )
            print("Saved epoch checkpoint to:", epoch_path)

        print(
            f"epoch {epoch:02d} | train loss: {train_loss:.5f} | val loss: {val_loss:.5f} "
            f"| val MS-SSIM: {val_score:.5f} (SSIM: {val_ssim:.5f}, PSNR: {val_psnr:.2f}, n={val_slices}) "
            f"| ms_ssim_w {ms_ssim_weight:.2f} l1_w {l1_weight:.2f}"
        )

        scheduler.step(val_score)

        if val_score > best_val:
            best_val = val_score
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": stage1.state_dict(),
                    "ema": ema.state_dict(),
                    "optim": optim1.state_dict(),
                    "val_loss": val_loss,
                    "val_score": val_score,
                    "val_ssim": val_ssim,
                    "val_ms_ssim": val_ms_ssim,
                    "val_psnr": val_psnr,
                },
                best_path
            )
            print("Saved best to:", best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping at epoch {epoch:02d} (best epoch {best_epoch:02d}, score {best_val:.5f})"
                )
                break

    best_overall_path = os.path.join(save_dir, "best_overall.ckpt")
    best_stage1_ckpt = torch.load(best_path, map_location=device)
    torch.save(
        {
            "stage": 1,
            "model": best_stage1_ckpt["model"],
            "ema": best_stage1_ckpt["ema"],
        },
        best_overall_path,
    )
    print("Saved overall best from stage1 to:", best_overall_path)
