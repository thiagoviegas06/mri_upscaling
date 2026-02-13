"""
STEP 2: Conditional Dataset for Diffusion Training

A diffusion model needs:
1. Input: low-field MRI (lf) - condition
2. Target: high-field MRI (hf) - what we're learning to denoise
3. Timestep: random timestep t (handled in training loop)

In training:
  condition = [lf, stage1_pred]  (2 channels)
  target = hf                     (1 channel)

We teach the model: given noisy target + condition, predict the noise.
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class ConditionalDiffusionDataset(Dataset):
    """
    Dataset for conditional diffusion training.
    
    Args:
        lf_paths: list of paths to low-field MRI files
        hf_paths: list of paths to high-field MRI files
        stage1_preds: list of stage1 predictions (or None to compute on-the-fly)
        patch_size: size of 3D patches to extract
        patches_per_volume: number of random patches per volume
        transform: optional augmentation
    """
    
    def __init__(self, lf_paths, hf_paths, stage1_preds=None, 
                 patch_size=96, patches_per_volume=32, transform=None):
        self.lf_paths = lf_paths
        self.hf_paths = hf_paths
        self.stage1_preds = stage1_preds
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.transform = transform
        
        # In this simple version, we'll compute stage1 preds on-the-fly
        # In practice, pre-compute or cache them for speed
        self.cache = {}
    
    def __len__(self):
        """Total number of patches."""
        return len(self.lf_paths) * self.patches_per_volume
    
    def _load_volume(self, path):
        """Load MRI volume from file."""
        # TODO: implement based on your file format
        # For now, return dummy data
        return np.random.randn(128, 128, 128).astype(np.float32)
    
    def _get_stage1_pred(self, lf):
        """
        In real code: run stage1 model on lf.
        For now: return dummy data (you'll replace this).
        """
        # Placeholder: in Step 4 (training), you'll use actual stage1 model
        return np.zeros_like(lf)
    
    def __getitem__(self, idx):
        """
        Get a single training example.
        
        Returns:
            condition: [lf, stage1_pred] concatenated
            target: high-field MRI
        """
        # Figure out which volume and which patch
        volume_idx = idx // self.patches_per_volume
        patch_idx = idx % self.patches_per_volume
        
        # Load volumes
        lf_vol = self._load_volume(self.lf_paths[volume_idx])  # (D, H, W)
        hf_vol = self._load_volume(self.hf_paths[volume_idx])  # (D, H, W)
        
        # Get stage1 prediction
        y1_vol = self._get_stage1_pred(lf_vol)  # (D, H, W)
        
        # Extract random patch
        patch = self._extract_random_patch(lf_vol, hf_vol, y1_vol)
        
        # Apply augmentation if provided
        if self.transform is not None:
            patch = self.transform(patch)
        
        return patch
    
    def _extract_random_patch(self, lf, hf, y1):
        """
        Extract a random 3D patch from volumes.
        
        Args:
            lf, hf, y1: (D, H, W) volumes
        
        Returns:
            {
                'condition': (2, D, H, W) - [lf, y1]
                'target': (1, D, H, W) - hf
            }
        """
        D, H, W = lf.shape
        p = self.patch_size
        
        # Random starting positions
        d_start = np.random.randint(0, max(1, D - p + 1))
        h_start = np.random.randint(0, max(1, H - p + 1))
        w_start = np.random.randint(0, max(1, W - p + 1))
        
        # Extract patches
        lf_patch = lf[d_start:d_start+p, h_start:h_start+p, w_start:w_start+p]
        hf_patch = hf[d_start:d_start+p, h_start:h_start+p, w_start:w_start+p]
        y1_patch = y1[d_start:d_start+p, h_start:h_start+p, w_start:w_start+p]
        
        # Concatenate condition
        condition = np.stack([lf_patch, y1_patch], axis=0).astype(np.float32)  # (2, D, H, W)
        target = hf_patch[np.newaxis, ...].astype(np.float32)  # (1, D, H, W)
        
        # Convert to torch
        return {
            'condition': torch.from_numpy(condition),
            'target': torch.from_numpy(target),
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("STEP 2: CONDITIONAL DATASET TUTORIAL")
    print("=" * 70)
    
    print("\n1. Dataset Structure:")
    print("-" * 70)
    print("Input to model:")
    print("  x_t: noisy target at timestep t         (1, D, H, W)")
    print("  condition: [lf, stage1_pred]            (2, D, H, W)")
    print("  t: timestep                             (scalar)")
    print("\nModel output:")
    print("  noise_pred: predicted noise             (1, D, H, W)")
    
    print("\n2. Loss Calculation:")
    print("-" * 70)
    print("During training:")
    print("  1. Sample random t")
    print("  2. Add noise to target: x_t = add_noise(hf, t, random_noise)")
    print("  3. Forward model: ε̂ = model(x_t, condition, t)")
    print("  4. Loss = MSE(ε̂, ε)")
    
    print("\n3. Data Flow:")
    print("-" * 70)
    print("""
    Volume level (128x128x128):
        lf_volume (low-field)      hf_volume (high-field)      stage1_pred
        (128, 128, 128)            (128, 128, 128)             (128, 128, 128)
               ↓                           ↓                           ↓
        Random 96x96x96 patch extraction
               ↓                           ↓                           ↓
        Batch dataset:
        - condition: (batch_size, 2, 96, 96, 96)  [lf_patch, y1_patch]
        - target:    (batch_size, 1, 96, 96, 96)  [hf_patch]
    """)
    
    print("\n4. Creating a Dataset Instance:")
    print("-" * 70)
    
    lf_files = ["dummy_lf_1", "dummy_lf_2"]
    hf_files = ["dummy_hf_1", "dummy_hf_2"]
    
    dataset = ConditionalDiffusionDataset(
        lf_paths=lf_files,
        hf_paths=hf_files,
        patch_size=96,
        patches_per_volume=32
    )
    
    print(f"Number of volumes: {len(lf_files)}")
    print(f"Patches per volume: 32")
    print(f"Total dataset size: {len(dataset)} patches")
    print(f"Each epoch trains on {len(dataset)} patches")
    
    print("\n5. In Training Loop:")
    print("-" * 70)
    print("""
    for epoch in range(num_epochs):
        for batch in dataloader:
            condition = batch['condition']  # (B, 2, D, H, W)
            target = batch['target']        # (B, 1, D, H, W)
            
            # Sample timestep
            t = torch.randint(0, 1000, (B,))  # random timestep
            
            # Add noise to target
            noise = torch.randn_like(target)
            x_t = schedule.add_noise(target, t, noise)
            
            # Model prediction
            noise_pred = model(x_t, condition, t)
            
            # Compute loss
            loss = F.mse_loss(noise_pred, noise)
            
            # Backward pass
            loss.backward()
            optim.step()
    """)
    
    print("\n" + "=" * 70)
    print("Next: Open 03_model.py to build the conditional model")
    print("=" * 70)
