import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from model import UNet3D
from refiner_model import RRDBNet
from preprocessing import load_pair_resample_normalize
from train import ssim_2d_metric

# --- Configuration ---
UNET_CHECKPOINT = "checkpoints/best.ckpt"
REFINER_CHECKPOINT = "checkpoints/refiner_best.ckpt"
BATCH_SIZE = 16
PATCH_SIZE = 96 # Size of 2D crops
NUM_EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SliceRefinementDataset(Dataset):
    def __init__(self, pairs, unet_model, patches_per_vol=20, device="cpu"):
        self.pairs = pairs
        self.unet = unet_model
        self.patches_per_vol = patches_per_vol
        self.device = device
        self.data_buffer = [] # Store (pred_slice, gt_slice) tuples

        print("Pre-generating refinement data from UNet...")
        self._generate_data()

    def _generate_data(self):
        self.unet.eval()
        with torch.no_grad():
            for lf_path, hf_path in self.pairs:
                # Load full volumes
                lf, hf = load_pair_resample_normalize(lf_path, hf_path, interp_order=1)
                
                # Run UNet on patches or full volume if it fits. 
                # For speed/memory, we'll extract random 3D patches, run UNet, then slice.
                D, H, W = lf.shape
                
                # Extract random crops
                for _ in range(self.patches_per_vol):
                    z = random.randint(0, D - PATCH_SIZE)
                    x = random.randint(0, H - PATCH_SIZE)
                    y = random.randint(0, W - PATCH_SIZE)
                    
                    lf_patch = lf[z:z+PATCH_SIZE, x:x+PATCH_SIZE, y:y+PATCH_SIZE]
                    hf_patch = hf[z:z+PATCH_SIZE, x:x+PATCH_SIZE, y:y+PATCH_SIZE]
                    
                    lf_t = torch.from_numpy(lf_patch)[None, None, ...].to(self.device)
                    pred_t = self.unet(lf_t)
                    pred_patch = pred_t.squeeze().cpu().numpy()
                    
                    # Convert 3D patch to 2D slices (Axial/Z-axis)
                    for k in range(pred_patch.shape[0]):
                        self.data_buffer.append((pred_patch[k], hf_patch[k]))

    def __len__(self):
        return len(self.data_buffer)

    def __getitem__(self, idx):
        pred, gt = self.data_buffer[idx]
        return torch.from_numpy(pred).unsqueeze(0), torch.from_numpy(gt).unsqueeze(0)

def main():
    # 1. Load UNet
    print(f"Loading UNet from {UNET_CHECKPOINT}")
    unet = UNet3D(base=56).to(DEVICE)
    ckpt = torch.load(UNET_CHECKPOINT, map_location=DEVICE)
    state = ckpt["ema"] if "ema" in ckpt else ckpt["model"]
    unet.load_state_dict(state)
    unet.requires_grad_(False) # Freeze UNet

    # 2. Prepare Data
    lf_dir = "mri_resolution/train/low_field"
    hf_dir = "mri_resolution/train/high_field"
    pairs = []
    for f in os.listdir(lf_dir):
        if f.endswith(".nii"):
            pairs.append((os.path.join(lf_dir, f), os.path.join(hf_dir, f.replace("lowfield", "highfield"))))
    
    # Use a subset for demo if dataset is huge
    dataset = SliceRefinementDataset(pairs[:20], unet, patches_per_vol=10, device=DEVICE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    # 3. Setup Refiner
    refiner = RRDBNet(in_nc=1, out_nc=1, nf=64, nb=4).to(DEVICE) # nb=4 for speed, use 23 for quality
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=1e-4)
    criterion = nn.L1Loss()
    scaler = GradScaler()

    # 4. Train Loop
    print(f"Starting training on {len(dataset)} slices...")
    for epoch in range(NUM_EPOCHS):
        refiner.train()
        epoch_loss = 0
        for pred_slice, gt_slice in loader:
            pred_slice, gt_slice = pred_slice.to(DEVICE), gt_slice.to(DEVICE)
            
            optimizer.zero_grad()
            with autocast():
                refined = refiner(pred_slice)
                loss = criterion(refined, gt_slice)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | L1 Loss: {avg_loss:.6f}")
        
        # Save checkpoint
        torch.save({"model": refiner.state_dict()}, REFINER_CHECKPOINT)

if __name__ == "__main__":
    main()