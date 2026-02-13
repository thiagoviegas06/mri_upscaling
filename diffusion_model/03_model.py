"""
STEP 3: Conditional UNet Model using HuggingFace Diffusers

The model needs to:
1. Accept noisy image x_t
2. Accept condition (lf, stage1_pred)
3. Accept timestep t
4. Output predicted noise

We'll use HuggingFace's built-in components for flexibility.
"""

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
import math


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding (like in transformers)."""
    
    def __init__(self, embedding_dim, freq_embed_dim=256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.freq_embed_dim = freq_embed_dim
        
        # MLP to project frequencies to embedding
        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
    
    def forward(self, timesteps):
        """
        Args:
            timesteps: (B,) tensor of timestep indices
        
        Returns:
            embeddings: (B, embedding_dim)
        """
        # Create sinusoidal embeddings (like positional encoding in transformers)
        half_dim = self.freq_embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        # Pass through MLP
        return self.mlp(emb)


class SimpleConditionalUNet3D(nn.Module):
    """
    Simple 3D UNet conditioned on:
    - Noisy input (x_t)
    - Condition (concatenated with x_t)
    - Timestep embedding (added at multiple levels)
    
    Architecture:
    - Encoder: downsample with skip connections
    - Bottleneck: process at lowest resolution
    - Decoder: upsample with skip connections
    - At each level: add timestep embedding
    
    This is inspired by DDPM/Stable Diffusion architectures.
    """
    
    def __init__(self, in_channels=1, cond_channels=2, out_channels=1, 
                 channels=(64, 128, 256, 512), num_res_blocks=2, 
                 attention_levels=None, embedding_dim=256):
        """
        Args:
            in_channels: channels in x_t (usually 1)
            cond_channels: channels in condition (usually 2: [lf, y1])
            out_channels: output channels (usually 1)
            channels: tuple of channel sizes at each level
            num_res_blocks: residual blocks per level
            attention_levels: which levels use self-attention (default: [3])
            embedding_dim: dimension of timestep embedding
        """
        super().__init__()
        
        if attention_levels is None:
            attention_levels = []
        
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.channels = channels
        self.num_levels = len(channels)
        self.embedding_dim = embedding_dim
        
        # Timestep embedding
        self.t_emb = TimestepEmbedding(embedding_dim)
        
        # Initial projection: concatenate x_t and condition
        self.input_proj = nn.Conv3d(
            in_channels + cond_channels, 
            channels[0], 
            kernel_size=3, 
            padding=1
        )
        
        # ====== ENCODER ======
        self.down_blocks = nn.ModuleList()
        for level in range(self.num_levels):
            out_ch = channels[level]
            self.down_blocks.append(
                ResidualBlock3D(
                    channels[level-1] if level > 0 else channels[0],
                    out_ch,
                    embedding_dim,
                    num_blocks=num_res_blocks,
                    use_attention=(level in attention_levels)
                )
            )
        
        # ====== BOTTLENECK ======
        self.middle_block = ResidualBlock3D(
            channels[-1], 
            channels[-1], 
            embedding_dim,
            num_blocks=num_res_blocks,
            use_attention=True
        )
        
        # ====== DECODER ======
        self.up_blocks = nn.ModuleList()
        for level in reversed(range(self.num_levels)):
            in_ch = channels[level] * 2 if level < self.num_levels - 1 else channels[level]
            self.up_blocks.append(
                ResidualBlock3D(
                    in_ch,
                    channels[level-1] if level > 0 else channels[0],
                    embedding_dim,
                    num_blocks=num_res_blocks,
                    use_attention=(level in attention_levels)
                )
            )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv3d(channels[0], out_channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x_t, condition, t):
        """
        Args:
            x_t: noisy image (B, 1, D, H, W)
            condition: [lf, y1] (B, 2, D, H, W)
            t: timesteps (B,)
        
        Returns:
            noise_pred: (B, 1, D, H, W)
        """
        # Timestep embedding
        t_emb = self.t_emb(t)  # (B, embedding_dim)
        
        # Concatenate input and condition
        x = torch.cat([x_t, condition], dim=1)  # (B, 3, D, H, W)
        
        # Project to initial channels
        x = self.input_proj(x)  # (B, channels[0], D, H, W)
        
        # Store skip connections
        skips = [x]
        
        # Encode (downsample)
        for down_block in self.down_blocks:
            x = down_block(x, t_emb)  # will implement downsampling
            skips.append(x)
        
        # Bottleneck
        x = self.middle_block(x, t_emb)
        
        # Decode (upsample)
        for up_block in self.up_blocks:
            # Concatenate skip connection (skip + current)
            x = torch.cat([x, skips.pop()], dim=1)
            x = up_block(x, t_emb)
        
        # Output
        noise_pred = self.output_proj(x)  # (B, 1, D, H, W)
        
        return noise_pred


class ResidualBlock3D(nn.Module):
    """3D Residual block with timestep conditioning."""
    
    def __init__(self, in_channels, out_channels, embedding_dim, 
                 num_blocks=2, use_attention=False):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                ResBlock3D(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    embedding_dim
                )
            )
        
        self.attention = None
        if use_attention:
            self.attention = AttentionBlock3D(out_channels)
    
    def forward(self, x, t_emb):
        for block in self.blocks:
            x = block(x, t_emb)
        
        if self.attention is not None:
            x = self.attention(x)
        
        return x


class ResBlock3D(nn.Module):
    """Single 3D residual block."""
    
    def __init__(self, in_channels, out_channels, embedding_dim):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        
        # Timestep embedding projection
        self.t_proj = nn.Linear(embedding_dim, out_channels)
        
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        
        # Skip projection if needed
        self.skip = None
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, 1)
    
    def forward(self, x, t_emb):
        # First conv + normalization
        h = self.norm1(x)
        h = nn.functional.silu(h)
        h = self.conv1(h)
        
        # Add timestep embedding
        t_emb_proj = self.t_proj(t_emb)  # (B, out_channels)
        h = h + t_emb_proj[:, :, None, None, None]  # broadcast to spatial dims
        
        # Second conv
        h = self.norm2(h)
        h = nn.functional.silu(h)
        h = self.conv2(h)
        
        # Skip connection
        if self.skip is not None:
            x = self.skip(x)
        
        return x + h


class AttentionBlock3D(nn.Module):
    """Self-attention block for 3D features."""
    
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
    
    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        
        # Reshape to (B*D*H*W, C) for attention
        x_norm = self.norm(x)
        x_flat = x_norm.permute(0, 2, 3, 4, 1).reshape(-1, C)  # (B*D*H*W, C)
        
        # Self-attention
        attn_out, _ = self.attn(x_flat.unsqueeze(1), x_flat.unsqueeze(1), x_flat.unsqueeze(1))
        attn_out = attn_out.squeeze(1)  # (B*D*H*W, C)
        
        # Reshape back
        attn_out = attn_out.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)
        
        return x + attn_out


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("STEP 3: CONDITIONAL UNET MODEL TUTORIAL")
    print("=" * 70)
    
    print("\n1. Model Architecture:")
    print("-" * 70)
    print("""
    Input:
      x_t: noisy image           (B, 1, 96, 96, 96)
      condition: [lf, y1]        (B, 2, 96, 96, 96)
      t: timestep                (B,)
    
    Processing:
      1. Timestep → embedding    (B, 256)
      2. Concatenate x_t + cond  (B, 3, 96, 96, 96)
      3. Project to base channels (B, 64, 96, 96, 96)
      4. Encoder (downsample):
         - 96 → 48 (with skip)
         - 48 → 24 (with skip)
         - 24 → 12 (with skip)
      5. Bottleneck at 12 resolution (with attention)
      6. Decoder (upsample):
         - 12 → 24 (concat skip)
         - 24 → 48 (concat skip)
         - 48 → 96 (concat skip)
      7. Output projection       (B, 1, 96, 96, 96)
    
    Output:
      noise_pred: predicted noise (B, 1, 96, 96, 96)
    """)
    
    print("\n2. Creating Model:")
    print("-" * 70)
    
    model = SimpleConditionalUNet3D(
        in_channels=1,
        cond_channels=2,
        out_channels=1,
        channels=(64, 128, 256),
        embedding_dim=256,
    )
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    print("\n3. Forward Pass Example:")
    print("-" * 70)
    
    device = 'cpu'
    model = model.to(device)
    
    # Create dummy inputs
    batch_size = 2
    x_t = torch.randn(batch_size, 1, 96, 96, 96, device=device)
    condition = torch.randn(batch_size, 2, 96, 96, 96, device=device)
    t = torch.tensor([100, 500], device=device)  # timesteps
    
    # Forward
    with torch.no_grad():
        noise_pred = model(x_t, condition, t)
    
    print(f"x_t shape:        {x_t.shape}")
    print(f"condition shape:  {condition.shape}")
    print(f"t shape:          {t.shape}")
    print(f"noise_pred shape: {noise_pred.shape}")
    
    print("\n4. Training One Step:")
    print("-" * 70)
    print("""
    import torch.nn.functional as F
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Target (high-field)
    target = torch.randn(batch_size, 1, 96, 96, 96, device=device)
    
    # Add noise (as in schedule.py)
    noise = torch.randn_like(target)
    x_t = schedule.add_noise(target, t, noise)
    
    # Predict
    noise_pred = model(x_t, condition, t)
    
    # Loss
    loss = F.mse_loss(noise_pred, noise)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    """)
    
    print("\n" + "=" * 70)
    print("Next: Open 04_train.py to implement the full training loop")
    print("=" * 70)
