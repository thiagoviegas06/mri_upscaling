import os
import random
import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from metric_model.model import UNet3D
from metric_model.train import train_one_epoch, validate, validate_metric
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
    optim = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scaler = GradScaler("cuda") if device == "cuda" else None

    best_val = float("-inf")
    save_dir = os.environ.get("MODEL_DESTINATION_METRIC", "checkpoints_metric")
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "best.ckpt")

    num_epochs = 50
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optim, device, scaler)
        val_loss = validate(model, val_loader, device)
        val_score, val_ssim, val_psnr, val_slices = validate_metric(
            model,
            val_pairs,
            device,
            patch_size=patch_size,
            stride=patch_size // 2,
        )

        if epoch % 5 == 0:
            epoch_path = os.path.join(save_dir, f"epoch_{epoch:02d}.ckpt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
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

        if val_score > best_val:
            best_val = val_score
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "val_loss": val_loss,
                    "val_score": val_score,
                    "val_ssim": val_ssim,
                    "val_psnr": val_psnr,
                },
                best_path
            )
            print("Saved best to:", best_path)
