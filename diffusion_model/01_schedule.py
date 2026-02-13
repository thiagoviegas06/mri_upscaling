"""
STEP 1: Understanding the Diffusion Schedule

A diffusion schedule defines how much noise to add/remove at each timestep.
This script helps you visualize and understand the schedule.

Run this to see:
- How alpha (signal) decreases over time
- How noise increases over time
- How to sample from the forward process
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


class LinearSchedule:
    """Linear noise schedule (simplest)."""
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=2e-2):
        self.timesteps = timesteps
        
        # Linear progression of noise variance
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        
        # Alpha = 1 - beta (how much signal to keep)
        self.alphas = 1.0 - self.betas
        
        # Cumulative product: α̅ₜ = ∏ αᵢ
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Precompute useful terms
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def add_noise(self, x0, t, noise):
        """
        Add noise to x0 according to the schedule.
        
        Forward process:
        x_t = sqrt(α̅ₜ) * x₀ + sqrt(1 - α̅ₜ) * ε
        
        Args:
            x0: original image (B, C, H, W) or (B, C, D, H, W)
            t: timestep(s), can be int or tensor of shape (B,)
            noise: random noise, same shape as x0
        
        Returns:
            x_t: noisy image
        """
        # Handle different input types
        if isinstance(t, int):
            t = torch.tensor([t])
        
        # Make sure t is 1D
        if len(t.shape) == 0:
            t = t.unsqueeze(0)
        
        # Get schedule values at time t
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting: (B,) → (B, 1, 1, 1, ...) or (B, 1, 1, 1)
        for _ in range(len(x0.shape) - 1):
            sqrt_alpha_t = sqrt_alpha_t.unsqueeze(-1)
            sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_t.unsqueeze(-1)
        
        # Apply forward process
        x_t = sqrt_alpha_t * x0 + sqrt_one_minus_alpha_t * noise
        return x_t


# ============================================================================
# TEST: Visualize the schedule
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("STEP 1: DIFFUSION SCHEDULE TUTORIAL")
    print("=" * 70)
    
    # Create schedule
    schedule = LinearSchedule(timesteps=1000, beta_start=1e-4, beta_end=2e-2)
    
    # Plot 1: Alpha and Beta over time
    print("\n1. Understanding Alpha and Beta:")
    print("-" * 70)
    
    t_samples = [0, 100, 250, 500, 750, 999]
    for t in t_samples:
        alpha_t = schedule.alphas_cumprod[t].item()
        noise_t = (1.0 - schedule.alphas_cumprod[t]).item()
        print(f"t={t:3d}: signal={alpha_t:.4f} ({alpha_t*100:.1f}%), "
              f"noise={noise_t:.4f} ({noise_t*100:.1f}%)")
    
    # Plot 2: Forward process on a simple image
    print("\n2. Forward Process (Adding Noise):")
    print("-" * 70)
    
    # Create a simple gradient image (0 to 1)
    x0 = torch.linspace(0, 1, 64).repeat(64, 1).unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64)
    print(f"Original image shape: {x0.shape}")
    print(f"Original image range: [{x0.min():.3f}, {x0.max():.3f}]")
    
    # Add noise at different timesteps
    noise = torch.randn_like(x0)
    print(f"\nAdding noise at different timesteps:")
    for t in [0, 250, 500, 750, 999]:
        x_t = schedule.add_noise(x0, t, noise)
        print(f"t={t:3d}: x_t range [{x_t.min():.3f}, {x_t.max():.3f}], "
              f"std={x_t.std():.3f}")
    
    # Plot 3: What does a timestep embedding look like?
    print("\n3. Timestep Embeddings (for the model):")
    print("-" * 70)
    print("The model needs to 'know' what timestep t it's at.")
    print("Usually done with sinusoidal embeddings (like in transformers):")
    print("\nExample sinusoidal embedding for t=100:")
    
    def sinusoidal_embedding(t, dim=256):
        """Create sinusoidal embedding for timestep."""
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = np.exp(np.arange(half_dim) * -emb)
        emb = t * emb
        emb = np.concatenate([np.sin(emb), np.cos(emb)])
        return emb
    
    t_emb = sinusoidal_embedding(100, dim=16)
    print(f"Shape: {t_emb.shape}")
    print(f"Values: {t_emb[:8]}")  # Print first 8
    
    # Plot 4: Key insight
    print("\n4. KEY INSIGHT:")
    print("-" * 70)
    print("At t=0 (start): x_t ≈ x0 (mostly signal, little noise)")
    print("At t=999 (end): x_t ≈ noise (almost pure noise)")
    print("\nDuring TRAINING:")
    print("  → Model learns to predict noise at each t")
    print("  → Loss = MSE(noise_predicted, noise_actual)")
    print("\nDuring INFERENCE (sampling):")
    print("  → Start from pure noise")
    print("  → Iteratively remove predicted noise")
    print("  → After 1000 steps: refined image")
    print("\nFaster inference:")
    print("  → Can use 50, 10, or even 4 steps instead of 1000")
    print("  → Trade-off: speed vs. quality")
    
    print("\n" + "=" * 70)
    print("Next: Open 02_dataset.py to learn how to prepare data")
    print("=" * 70)
