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

from metric_model.model import RefinerUNet3D, UNet3D
from metric_model.train import EMA, train_one_epoch, train_one_epoch_refiner, validate, validate_metric
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

    pairs = make_pairs("/scratch/tjv235/pytorch-example/mri_upscaling/train/low_field", "/scratch/tjv235/pytorch-example/mri_upscaling/train/high_field")
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
        augment=True,
    )
    val_ds   = MRIPatchDataset(val_pairs,   patch_size=patch_size, patches_per_volume=16, cache_volumes=True)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=2, shuffle=False, num_workers=2, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stage1 = UNet3D(base=56).to(device)
    refiner = RefinerUNet3D(in_ch=2, out_ch=1, base=24, dropout_p=0.0).to(device)

    optim1 = torch.optim.AdamW(stage1.parameters(), lr=2e-4, weight_decay=1e-4)
    optim2 = torch.optim.AdamW(refiner.parameters(), lr=1e-4, weight_decay=1e-4)
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

    num_epochs = 60
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(stage1, train_loader, optim1, device, scaler, ema=ema)

        ema_backup = ema.apply_to(stage1)
        val_loss = validate(stage1, val_loader, device)
        val_score, val_ssim, val_psnr, val_slices = validate_metric(
            stage1,
            val_pairs,
            device,
            patch_size=patch_size,
            stride=patch_size // 2,
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
                    "val_psnr": val_psnr,
                },
                epoch_path
            )
            print("Saved epoch checkpoint to:", epoch_path)

        print(
            f"epoch {epoch:02d} | train loss: {train_loss:.5f} | val loss: {val_loss:.5f} "
            f"| val score: {val_score:.5f} (ssim {val_ssim:.5f}, psnr {val_psnr:.2f}, n={val_slices})"
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

    # --- Load best stage1 before training stage2 ---
    best_ckpt = torch.load(best_path, map_location=device)
    stage1.load_state_dict(best_ckpt["model"])
    ema.load_state_dict(best_ckpt["ema"])
    print(f"Loaded best stage1 from {best_path} (epoch {best_ckpt.get('epoch')})")

    for p in stage1.parameters():
        p.requires_grad = False
    stage1.eval()

    num_epochs_stage2 = 30
    best_val2 = float("-inf")
    best_epoch2 = 0
    epochs_no_improve2 = 0
    patience2 = 10
    best_refiner_path = os.path.join(save_dir, "best_refiner.ckpt")

    for epoch in range(1, num_epochs_stage2 + 1):
        train_loss = train_one_epoch_refiner(
            stage1,
            ema,
            refiner,
            train_loader,
            optim2,
            device,
            scaler=scaler,
            delta_l1_weight=0.01,
        )
        val_score, val_ssim, val_psnr, val_slices = validate_metric(
            stage1,
            val_pairs,
            device,
            ema=ema,
            refiner=refiner,
            patch_size=patch_size,
            stride=patch_size // 2,
        )

        if epoch % 5 == 0:
            epoch_path = os.path.join(save_dir, f"refiner_epoch_{epoch:02d}.ckpt")
            torch.save(
                {
                    "epoch": epoch,
                    "stage1": stage1.state_dict(),
                    "refiner": refiner.state_dict(),
                    "ema": ema.state_dict(),
                    "optim": optim2.state_dict(),
                    "val_score": val_score,
                    "val_ssim": val_ssim,
                    "val_psnr": val_psnr,
                },
                epoch_path,
            )
            print("Saved refiner checkpoint to:", epoch_path)

        print(
            f"stage2 epoch {epoch:02d} | train loss: {train_loss:.5f} | val score: {val_score:.5f} "
            f"(ssim {val_ssim:.5f}, psnr {val_psnr:.2f}, n={val_slices})"
        )

        if val_score > best_val2:
            best_val2 = val_score
            best_epoch2 = epoch
            epochs_no_improve2 = 0
            torch.save(
                {
                    "epoch": epoch,
                    "stage1": stage1.state_dict(),
                    "refiner": refiner.state_dict(),
                    "ema": ema.state_dict(),
                    "optim": optim2.state_dict(),
                    "val_score": val_score,
                    "val_ssim": val_ssim,
                    "val_psnr": val_psnr,
                },
                best_refiner_path,
            )
            print("Saved best refiner to:", best_refiner_path)
        else:
            epochs_no_improve2 += 1
            if epochs_no_improve2 >= patience2:
                print(
                    f"Stage2 early stopping at epoch {epoch:02d} (best epoch {best_epoch2:02d}, "
                    f"score {best_val2:.5f})"
                )
                break

    best_overall_path = os.path.join(save_dir, "best_overall.ckpt")
    if best_val2 > best_val:
        best_refiner_ckpt = torch.load(best_refiner_path, map_location=device)
        torch.save(
            {
                "stage": 2,
                "model": best_refiner_ckpt["stage1"],
                "ema": best_refiner_ckpt["ema"],
                "refiner": best_refiner_ckpt["refiner"],
            },
            best_overall_path,
        )
        print("Saved overall best from stage2 to:", best_overall_path)
    else:
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
