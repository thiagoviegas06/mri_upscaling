import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Import existing pipeline components
from main import make_pairs, split_pairs
from model import UNet3D
from train import validate, validate_metric, EMA
from preprocessing import load_pair_resample_normalize, MRIPatchDataset
from refiner_model import GrayscaleRealESRGAN_1x, CascadedModel

# --- Configuration ---
UNET_CHECKPOINT = "checkpoints/best.ckpt"
REFINER_CHECKPOINT = "checkpoints/refiner_best.ckpt"
REALESRGAN_PATH = "/content/RealESRGAN_x4plus.pth"
BATCH_SIZE = 16
PATCH_SIZE = 96
NUM_EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SliceRefinementDataset(Dataset):
    """
    Generates (UNet_Prediction_Slice, Ground_Truth_Slice) pairs 
    from the TRAIN split only.
    """
    def __init__(self, pairs, unet_model, patches_per_vol=20, device="cpu"):
        self.data_buffer = []
        unet_model.eval()
        
        print(f"Generating refinement data from {len(pairs)} training volumes...")
        
        with torch.no_grad():
            for lf_path, hf_path in pairs:
                lf, hf = load_pair_resample_normalize(lf_path, hf_path, interp_order=1)
                D, H, W = lf.shape
                
                # Extract random 3D patches from the volume
                for _ in range(patches_per_vol):
                    z = random.randint(0, D - PATCH_SIZE)
                    x = random.randint(0, H - PATCH_SIZE)
                    y = random.randint(0, W - PATCH_SIZE)
                    
                    lf_patch = lf[z:z+PATCH_SIZE, x:x+PATCH_SIZE, y:y+PATCH_SIZE]
                    hf_patch = hf[z:z+PATCH_SIZE, x:x+PATCH_SIZE, y:y+PATCH_SIZE]
                    
                    # Run UNet on the patch
                    lf_t = torch.from_numpy(lf_patch)[None, None, ...].to(device)
                    pred_t = unet_model(lf_t)
                    pred_patch = pred_t.squeeze().cpu().numpy() # (D, H, W)
                    
                    # Slice the 3D patch into 2D samples for the Refiner
                    # (We only take the middle slices to avoid boundary artifacts if desired, 
                    # but taking all is fine for efficiency)
                    for k in range(pred_patch.shape[0]):
                        self.data_buffer.append((pred_patch[k], hf_patch[k]))

    def __len__(self):
        return len(self.data_buffer)

    def __getitem__(self, idx):
        pred, gt = self.data_buffer[idx]
        # Add channel dimension (1, H, W)
        return torch.from_numpy(pred).unsqueeze(0), torch.from_numpy(gt).unsqueeze(0)

def train_refiner_one_epoch(model, loader, optim, scaler, device):
    model.train()
    running_loss = 0.0
    criterion = nn.MSELoss()
    
    for pred_slice, hf_slice in loader:
        pred_slice = pred_slice.to(device, non_blocking=True)
        hf_slice = hf_slice.to(device, non_blocking=True)
        
        optim.zero_grad()
        
        with autocast():
            # Refiner input is the UNet output (pred_slice)
            refined = model(pred_slice)
            loss = criterion(refined, hf_slice)
            
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        
        running_loss += loss.item()
        
    return running_loss / len(loader)

def main():
    # 1. Setup Data Splits (Matching train.py)
    full_pairs = make_pairs("mri_resolution/train/low_field", "mri_resolution/train/high_field")
    train_pairs, val_pairs = split_pairs(full_pairs, val_frac=0.2, seed=42)
    
    print(f"Split: {len(train_pairs)} Train, {len(val_pairs)} Val")

    # 2. Load Pre-trained UNet
    print(f"Loading UNet from {UNET_CHECKPOINT}...")
    unet = UNet3D(base=56).to(DEVICE)
    ckpt = torch.load(UNET_CHECKPOINT, map_location=DEVICE)
    state = ckpt["ema"] if "ema" in ckpt else ckpt["model"]
    unet.load_state_dict(state)
    unet.requires_grad_(False)
    unet.eval()

    # 3. Prepare Datasets
    # Training: 2D Slices (pre-generated from UNet output)
    train_ds = SliceRefinementDataset(train_pairs, unet, patches_per_vol=16, device=DEVICE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    
    # Validation: 3D Patches (standard pipeline validation)
    # We use the existing MRIPatchDataset for fair comparison
    val_ds = MRIPatchDataset(val_pairs, patch_size=PATCH_SIZE, patches_per_volume=16, cache_volumes=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, pin_memory=True)

    # 4. Setup Refiner
    print(f"Loading Real-ESRGAN weights from {REALESRGAN_PATH}...")
    
    # OLD
    # refiner = RRDBNet(in_nc=1, out_nc=1, nf=64, nb=4).to(DEVICE)
    
    # NEW
    # This automatically loads the pretrained weights and performs the 1x/1-channel surgery
    refiner = GrayscaleRealESRGAN_1x(model_path=REALESRGAN_PATH, device=DEVICE).to(DEVICE)
    
    # Note: We use a smaller learning rate because the body is already pretrained
    # You might want to lower this from 5e-4 to 1e-4 or 5e-5 to preserve features
    optim = torch.optim.AdamW(refiner.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # 5. Add a Scheduler (same as in your original train.py)
    # This protects you: if 5e-4 is too high and loss jumps, it will drop the LR.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode="max",      # We want to maximize 'val_score'
        factor=0.5,      # Cut LR in half if stuck
        patience=3,      # Wait 3 epochs before cutting
        min_lr=1e-6
    )
    scaler = GradScaler()
    
    # Optional: EMA for Refiner
    # ema = EMA(refiner, decay=0.999)

    # 6. Cascaded Model for Validation
    # This wraps (UNet + Refiner) to look like a single 3D model to the validator
    cascaded_model = CascadedModel(unet, refiner, device=DEVICE)

    best_val_score = float("-inf")

    print("Starting Refiner Training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        # -- Train Step (2D) --
        train_loss = train_refiner_one_epoch(refiner, train_loader, optim, scaler, DEVICE)
        # ema.update(refiner)
        
        # -- Validation Step (3D) --
        # We use the EMA weights for validation
        # ema_backup = ema.apply_to(refiner)
        
        # A. Calculate Val L1 Loss (and 3D SSIM loss component) using `validate` from train.py
        val_loss = validate(cascaded_model, val_loader, DEVICE)
        
        # B. Calculate Full Metrics (Score, SSIM, PSNR) using `validate_metric` from train.py
        # This runs on full volumes (or slices of them)
        val_score, val_ssim, val_psnr, val_slices = validate_metric(
            cascaded_model, 
            val_pairs, 
            DEVICE, 
            patch_size=PATCH_SIZE, 
            stride=PATCH_SIZE // 2
        )
        
        scheduler.step(val_score)
        # ema.restore(refiner, ema_backup)

        # -- Logging --
        print(
            f"epoch {epoch:02d} | train L1: {train_loss:.5f} | val L1: {val_loss:.5f} "
            f"| val score: {val_score:.5f} (ssim {val_ssim:.5f}, psnr {val_psnr:.2f}, n={val_slices})"
        )

        # -- Checkpointing --
        if val_score > best_val_score:
            best_val_score = val_score
            torch.save({
                "epoch": epoch,
                "model": refiner.state_dict(),
                # "ema": ema.state_dict(),
                "val_score": val_score
            }, REFINER_CHECKPOINT)
            print(f"Saved Refiner Best: {REFINER_CHECKPOINT}")

if __name__ == "__main__":
    main()