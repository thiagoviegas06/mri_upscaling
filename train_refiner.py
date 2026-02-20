import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from main import make_pairs, split_pairs
from model import UNet3D
from train import validate, validate_metric
from preprocessing import load_pair_resample_normalize, MRIPatchDataset
from loss import VGGPerceptualLoss

# --- NEW IMPORTS ---
from refiner_model import SMPRefiner, CascadedModel

UNET_CHECKPOINT = "checkpoints/best.ckpt"
REFINER_CHECKPOINT = "checkpoints/refiner_best.ckpt"
BATCH_SIZE = 16
PATCH_SIZE = 96
NUM_EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SliceRefinementDataset(Dataset):
    def __init__(self, pairs, unet_model, patches_per_vol=20, device="cpu"):
        self.data_buffer = []
        unet_model.eval()
        print(f"Generating refinement data from {len(pairs)} training volumes...")
        with torch.no_grad():
            for lf_path, hf_path in pairs:
                lf, hf = load_pair_resample_normalize(lf_path, hf_path, interp_order=1)
                D, H, W = lf.shape
                for _ in range(patches_per_vol):
                    z = random.randint(0, D - PATCH_SIZE)
                    x = random.randint(0, H - PATCH_SIZE)
                    y = random.randint(0, W - PATCH_SIZE)
                    lf_patch = lf[z:z+PATCH_SIZE, x:x+PATCH_SIZE, y:y+PATCH_SIZE]
                    hf_patch = hf[z:z+PATCH_SIZE, x:x+PATCH_SIZE, y:y+PATCH_SIZE]
                    lf_t = torch.from_numpy(lf_patch)[None, None, ...].to(device)
                    pred_t = unet_model(lf_t)
                    pred_patch = pred_t.squeeze().cpu().numpy()
                    for k in range(pred_patch.shape[0]):
                        self.data_buffer.append((pred_patch[k], hf_patch[k]))

    def __len__(self):
        return len(self.data_buffer)

    def __getitem__(self, idx):
        pred, gt = self.data_buffer[idx]
        return torch.from_numpy(pred).unsqueeze(0), torch.from_numpy(gt).unsqueeze(0)

def train_refiner_one_epoch(model, loader, optim, scaler, criterion, device):
    model.train()
    running_loss = 0.0
    for pred_slice, hf_slice in loader:
        pred_slice = pred_slice.to(device, non_blocking=True)
        hf_slice = hf_slice.to(device, non_blocking=True)
        
        optim.zero_grad()
        with autocast():
            refined = model(pred_slice)
            loss = criterion(refined, hf_slice)
            
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        running_loss += loss.item()
    return running_loss / len(loader)

def main():
    full_pairs = make_pairs("mri_resolution/train/low_field", "mri_resolution/train/high_field")
    train_pairs, val_pairs = split_pairs(full_pairs, val_frac=0.2, seed=42)

    print(f"Loading UNet from {UNET_CHECKPOINT}...")
    # Make sure to load with base=16 to match the simplified architecture
    unet = UNet3D(base=16).to(DEVICE)
    ckpt = torch.load(UNET_CHECKPOINT, map_location=DEVICE)
    state = ckpt["ema"] if "ema" in ckpt else ckpt["model"]
    unet.load_state_dict(state)
    unet.requires_grad_(False)
    unet.eval()

    train_ds = SliceRefinementDataset(train_pairs, unet, patches_per_vol=16, device=DEVICE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_ds = MRIPatchDataset(val_pairs, patch_size=PATCH_SIZE, patches_per_volume=16, cache_volumes=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, pin_memory=True)

    print("Loading pre-trained MobileNetV2 UNet Refiner...")
    refiner = SMPRefiner(encoder_name="mobilenet_v2", encoder_weights="imagenet").to(DEVICE)
    
    # Restored learning rate to 1e-3 for a simpler architecture
    optim = torch.optim.AdamW(refiner.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = VGGPerceptualLoss(use_l1=True).to(DEVICE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="max", factor=0.5, patience=3, min_lr=1e-6)
    scaler = GradScaler()
    cascaded_model = CascadedModel(unet, refiner, device=DEVICE)

    best_val_score = float("-inf")

    print("Starting Refiner Training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_refiner_one_epoch(refiner, train_loader, optim, scaler, criterion, DEVICE)
        val_loss = validate(cascaded_model, val_loader, DEVICE)
        val_score, val_ssim, val_psnr, val_slices = validate_metric(
            cascaded_model, val_pairs, DEVICE, patch_size=PATCH_SIZE, stride=PATCH_SIZE // 2
        )
        
        scheduler.step(val_score)

        print(
            f"epoch {epoch:02d} | train Loss: {train_loss:.5f} | val L1: {val_loss:.5f} "
            f"| val score: {val_score:.5f} (ssim {val_ssim:.5f}, psnr {val_psnr:.2f})"
        )

        if val_score > best_val_score:
            best_val_score = val_score
            torch.save({
                "epoch": epoch,
                "model": refiner.state_dict(),
                "val_score": val_score
            }, REFINER_CHECKPOINT)
            print(f"Saved Refiner Best: {REFINER_CHECKPOINT}")

if __name__ == "__main__":
    main()