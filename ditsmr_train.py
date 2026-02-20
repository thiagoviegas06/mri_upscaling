import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast # Import for Mixed Precision
from preprocessing import MRIPatchDataset
from ditsmr import DiTMSR

# ==========================================
# Helpers
# ==========================================

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lf_dir', type=str, required=True, help='Path to 64mT NifTI files')
    parser.add_argument('--hf_dir', type=str, required=True, help='Path to 3T NifTI files')
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2], help='1: Pretrain AE, 2: Train Diffusion')
    parser.add_argument('--epochs', type=int, default=100)
    # Reduced default batch size to 2 to be safe, rely on micro_batching for memory management
    parser.add_argument('--batch_size', type=int, default=2) 
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patch_size', type=int, default=64)
    # New arg for memory control
    parser.add_argument('--micro_batch_size', type=int, default=16, help='Number of slices processed in one forward pass to save memory')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    return parser.parse_args()

def frequency_loss(pred, target):
    """
    Computes L2 loss in K-Space (Fourier Domain).
    """
    # FFT must be done in float32 for stability, autocast might cast to fp16 otherwise
    fft_pred = torch.fft.fft2(pred.float())
    fft_target = torch.fft.fft2(target.float())
    
    loss = torch.norm(fft_pred - fft_target, p=2) / pred.numel()
    return loss

def collate_patches(batch):
    """
    Batch is list of (LF, HF) tensors.
    LF/HF shape: (1, PatchSize, PatchSize, PatchSize) -> 3D
    """
    # item[0] is (1, X, Y, Z), cat dim=0 gives (B, X, Y, Z)
    lfs = torch.cat([item[0] for item in batch], dim=0) 
    hfs = torch.cat([item[1] for item in batch], dim=0)
    
    # Add Channel dim -> (B, 1, X, Y, Z)
    lfs = lfs.unsqueeze(1)
    hfs = hfs.unsqueeze(1)
    return lfs, hfs

# ==========================================
# Main Training Loops
# ==========================================

def train_stage1(model, loader, optimizer, device, epochs, save_dir, micro_batch_size):
    """
    Stage 1: Train Autoencoder with Gradient Accumulation (Micro-batching)
    """
    criterion_l1 = nn.L1Loss()
    scaler = GradScaler() # For Mixed Precision
    
    print(f"Starting Stage 1: Autoencoder Training (Micro-batch: {micro_batch_size})...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for i, (lf, hf) in enumerate(loader):
            # Input: (B, 1, D, H, W)
            # Reshape to 2D slices: (B*D, 1, H, W)
            b, c, d, h, w = lf.shape
            lf_flat = lf.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
            hf_flat = hf.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
            
            # Shuffle slices to break volume coherence slightly (optional, but good for stability)
            perm = torch.randperm(lf_flat.size(0))
            lf_flat = lf_flat[perm]
            hf_flat = hf_flat[perm]

            num_slices = lf_flat.size(0)
            optimizer.zero_grad()
            
            # --- Micro-batch Loop ---
            # Process the large batch in small chunks to fit in VRAM
            batch_loss = 0
            for start_idx in range(0, num_slices, micro_batch_size):
                end_idx = min(start_idx + micro_batch_size, num_slices)
                
                # Move only the micro-batch to GPU
                lf_micro = lf_flat[start_idx:end_idx].to(device)
                hf_micro = hf_flat[start_idx:end_idx].to(device)
                
                # Mixed Precision Context
                with autocast():
                    sr = model.forward_stage1(lf_micro, lf_micro)
                    loss_pixel = criterion_l1(sr, hf_micro)
                    loss_freq = frequency_loss(sr, hf_micro)
                    loss = loss_pixel + 0.1 * loss_freq
                    # Scale loss by number of micro-batches to keep gradient magnitude consistent
                    # (Simple averaging logic)
                    loss = loss / (num_slices / micro_batch_size) 
                
                # Backward pass with Scaler
                scaler.scale(loss).backward()
                
                # Detach to save memory and accumulate scalar for print
                batch_loss += loss.item() * (num_slices / micro_batch_size)

            # Update weights after processing all micro-batches
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += batch_loss
            
            if i % 10 == 0:
                print(f"Epoch {epoch} [{i}/{len(loader)}] Loss: {batch_loss:.4f}")
        
        print(f"Epoch {epoch} Avg Loss: {epoch_loss / len(loader):.4f}")
        
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'stage1_epoch_{epoch+1}.pth'))

def train_stage2(model, loader, optimizer, device, epochs, save_dir, micro_batch_size):
    """
    Stage 2: Diffusion Training with Micro-batching
    """
    print(f"Starting Stage 2: Diffusion Training (Micro-batch: {micro_batch_size})...")
    
    # Freeze Encoders
    for param in model.lr_encoder.parameters():
        param.requires_grad = False
    for param in model.ref_encoder.parameters():
        param.requires_grad = False
        
    model.set_noise_schedule(num_steps=1000)
    scaler = GradScaler()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for i, (lf, hf) in enumerate(loader):
            b, c, d, h, w = lf.shape
            lf_flat = lf.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
            hf_flat = hf.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
            
            # Sample timestep for the WHOLE batch first (or per slice, let's do per slice for diversity)
            t_all = torch.randint(0, 1000, (lf_flat.size(0),), device=device).long()
            
            num_slices = lf_flat.size(0)
            optimizer.zero_grad()
            
            batch_loss_diff = 0
            batch_loss_recon = 0
            
            # --- Micro-batch Loop ---
            for start_idx in range(0, num_slices, micro_batch_size):
                end_idx = min(start_idx + micro_batch_size, num_slices)
                
                lf_micro = lf_flat[start_idx:end_idx].to(device)
                hf_micro = hf_flat[start_idx:end_idx].to(device)
                t_micro = t_all[start_idx:end_idx] # t is already on device
                
                with autocast():
                    # 1. Diffusion Loss
                    # HF is x0 target, LF is condition
                    pred_noise, noise = model.forward_diffusion_train(hf_micro, lf_micro, t_micro)
                    loss_diff = nn.MSELoss()(pred_noise, noise)
                    
                    # 2. Reconstruction Loss
                    # We usually train reconstruction on the clean path or estimated x0
                    # Here we follow standard strategy: Train decoder to map encoded HF latent to HF image
                    # Need to re-encode HF/LF for this part (activations not retained from diff step)
                    with torch.no_grad():
                        hf_latent = model.lr_encoder(hf_micro)
                    
                    lf_feat = model.lr_encoder(lf_micro)
                    sr_recon = model.decoder(hf_latent, lf_feat)
                    
                    loss_recon_px = nn.L1Loss()(sr_recon, hf_micro)
                    loss_recon_freq = frequency_loss(sr_recon, hf_micro)
                    loss_recon = loss_recon_px + 0.1 * loss_recon_freq
                    
                    total_loss = loss_diff + loss_recon
                    # Scale for accumulation
                    total_loss = total_loss / (num_slices / micro_batch_size)

                scaler.scale(total_loss).backward()
                
                # Stats
                batch_loss_diff += loss_diff.item()
                batch_loss_recon += loss_recon.item()

            scaler.step(optimizer)
            scaler.update()
            
            # Average out the losses for printing
            avg_diff = batch_loss_diff / (num_slices / micro_batch_size)
            avg_recon = batch_loss_recon / (num_slices / micro_batch_size)
            
            epoch_loss += (avg_diff + avg_recon)
            
            if i % 10 == 0:
                print(f"Epoch {epoch} [{i}/{len(loader)}] Diff: {avg_diff:.4f} Recon: {avg_recon:.4f}")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'stage2_epoch_{epoch+1}.pth'))

# ==========================================
# Main
# ==========================================

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    lf_files = sorted([os.path.join(args.lf_dir, f) for f in os.listdir(args.lf_dir)])
    hf_files = sorted([os.path.join(args.hf_dir, f) for f in os.listdir(args.hf_dir)])
    
    # Validation
    if len(lf_files) == 0:
        raise ValueError(f"No files found in {args.lf_dir}")
    if len(lf_files) != len(hf_files):
        print(f"Warning: Number of LF files ({len(lf_files)}) != HF files ({len(hf_files)})")
        # Truncate to shorter
        min_len = min(len(lf_files), len(hf_files))
        lf_files = lf_files[:min_len]
        hf_files = hf_files[:min_len]

    pairs = list(zip(lf_files, hf_files))
    
    # 32 patches per volume * batch size 2 = 64 patches per step.
    # 64 patches * 64 depth = 4096 slices total per step.
    # Processing 4096 slices at once is impossible.
    # Micro-batching of 16 means 256 micro-steps. This will work on 40GB easily.
    dataset = MRIPatchDataset(
        pairs, 
        patch_size=args.patch_size, 
        patches_per_volume=32, 
        cache_volumes=False 
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_patches,
        num_workers=4,
        pin_memory=True
    )
    
    model = DiTMSR(device=device).to(device)
    
    if args.checkpoint:
        print(f"Loading checkpoint {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint))
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    if args.stage == 1:
        train_stage1(model, loader, optimizer, device, args.epochs, args.save_dir, args.micro_batch_size)
    elif args.stage == 2:
        train_stage2(model, loader, optimizer, device, args.epochs, args.save_dir, args.micro_batch_size)

if __name__ == '__main__':
    main()