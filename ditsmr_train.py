import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    return parser.parse_args()

def frequency_loss(pred, target):
    """
    Computes L2 loss in K-Space (Fourier Domain).
    As per paper: ||K_DC - K_HR||_2
    """
    fft_pred = torch.fft.fft2(pred.float())
    fft_target = torch.fft.fft2(target.float())
    
    # We can use the magnitude or the full complex difference
    loss = torch.norm(fft_pred - fft_target, p=2) / pred.numel()
    return loss

def collate_patches(batch):
    """
    Batch is list of (LF, HF) tensors.
    LF/HF shape: (1, PatchSize, PatchSize, PatchSize) -> 3D
    We need to stack them.
    """
    lfs = torch.cat([item[0] for item in batch], dim=0) # (B, X, Y, Z)
    hfs = torch.cat([item[1] for item in batch], dim=0)
    
    # Add Channel dim (B, C, X, Y, Z)
    lfs = lfs.unsqueeze(1)
    hfs = hfs.unsqueeze(1)
    return lfs, hfs

# ==========================================
# Main Training Loops
# ==========================================

def train_stage1(model, loader, optimizer, device, epochs, save_dir):
    """
    Stage 1: Train Encoders and Decoder using L1 + Frequency Loss.
    This sets up the Latent Space and the Reconstruction capability.
    """
    criterion_l1 = nn.L1Loss()
    
    print("Starting Stage 1: Autoencoder Training...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for i, (lf, hf) in enumerate(loader):
            # Input: (B, 1, D, H, W). Paper uses 2D.
            # We reshape to (B*D, 1, H, W) to treat slices as batch items
            b, c, d, h, w = lf.shape
            lf = lf.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w).to(device)
            hf = hf.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w).to(device)
            
            optimizer.zero_grad()
            
            # Forward (Use Encoder -> Decoder directly)
            # In Stage 1, we use LF as both input and 'reference' structure for itself
            sr = model.forward_stage1(lf, lf)
            
            loss_pixel = criterion_l1(sr, hf)
            loss_freq = frequency_loss(sr, hf)
            
            loss = loss_pixel + 0.1 * loss_freq # Weighting from paper/heuristics
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch {epoch} [{i}/{len(loader)}] Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch} Avg Loss: {epoch_loss / len(loader):.4f}")
        
        # Save
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'stage1_epoch_{epoch+1}.pth'))

def train_stage2(model, loader, optimizer, device, epochs, save_dir):
    """
    Stage 2: Train Diffusion Transformer (DiT) and fine-tune Decoder.
    Freeze Encoders (mostly).
    """
    print("Starting Stage 2: Diffusion Training...")
    
    # Freeze Encoders
    for param in model.lr_encoder.parameters():
        param.requires_grad = False
    for param in model.ref_encoder.parameters():
        param.requires_grad = False
        
    model.set_noise_schedule(num_steps=1000)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for i, (lf, hf) in enumerate(loader):
            # Reshape 3D -> 2D slices
            b, c, d, h, w = lf.shape
            lf = lf.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w).to(device)
            hf = hf.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w).to(device)
            
            optimizer.zero_grad()
            
            # Sample timestep
            t = torch.randint(0, 1000, (lf.shape[0],), device=device).long()
            
            # 1. Diffusion Loss
            # Input: LR (as conditioning context for diffusion? No, paper uses Ref)
            # Paper: LR -> Latent. Noise added to Latent. Ref -> Condition.
            # Since we only have LF/HF pairs, we treat LF as "Target" (Noisy) source and also as Condition?
            # Actually, standard SR Diffusion: 
            # Condition = LR Image. Target = HR Latent.
            # Paper DiTMSR: Condition = Reference HR. Target = LR Latent (Denoising LR to HR? No).
            # Clarification: Super Resolution generates HR.
            # x0 = HR Latent. Condition = LR Latent.
            # Let's map this: x0 comes from HF. Condition comes from LF.
            
            # CORRECT MAPPING for SR:
            # We want to generate HF. So x0 is derived from HF.
            # We condition on LF.
            
            # In forward_diffusion_train:
            # x0 = self.encode(hf) (We want to learn distribution of HF latents)
            # condition = self.encode(lf) (Guided by LF)
            
            # Note: The model.py `forward_diffusion_train` expects inputs.
            # We pass HF as the target to be noised, LF as reference/condition.
            pred_noise, noise = model.forward_diffusion_train(hf, lf, t)
            
            loss_diff = nn.MSELoss()(pred_noise, noise)
            
            # 2. Reconstruction Loss (Decoder Training)
            # We also pass the Clean HF latent through decoder to ensure it maps to HF Image
            # (Optional in standard LDM, but paper implies decoder is trained here too)
            with torch.no_grad():
                hf_latent = model.lr_encoder(hf) # Reuse LR encoder weights or HF specific? 
                # Ideally we use an HR encoder for GT, but let's use the trained encoder.
            
            # Refine Decoder
            lf_feat = model.lr_encoder(lf)
            sr_recon = model.decoder(hf_latent, lf_feat)
            loss_recon = nn.L1Loss()(sr_recon, hf) + 0.1 * frequency_loss(sr_recon, hf)
            
            loss = loss_diff + loss_recon
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch {epoch} [{i}/{len(loader)}] Diff: {loss_diff.item():.4f} Recon: {loss_recon.item():.4f}")

        # Save
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
    
    # Data Setup
    # Assumes file pairs are lists of strings. 
    # You need to implement a function to list your .nii.gz files from args.lf_dir / args.hf_dir
    # For now, placeholder:
    lf_files = sorted([os.path.join(args.lf_dir, f) for f in os.listdir(args.lf_dir)])
    hf_files = sorted([os.path.join(args.hf_dir, f) for f in os.listdir(args.hf_dir)])
    pairs = list(zip(lf_files, hf_files))
    
    dataset = MRIPatchDataset(
        pairs, 
        patch_size=args.patch_size, 
        patches_per_volume=32, 
        cache_volumes=False # Turn off if memory is tight
    )
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_patches)
    
    # Model Setup
    model = DiTMSR(device=device).to(device)
    
    if args.checkpoint:
        print(f"Loading checkpoint {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint))
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    if args.stage == 1:
        train_stage1(model, loader, optimizer, device, args.epochs, args.save_dir)
    elif args.stage == 2:
        train_stage2(model, loader, optimizer, device, args.epochs, args.save_dir)

if __name__ == '__main__':
    main()