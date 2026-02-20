import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Try importing Mamba, otherwise fall back to a PyTorch approximation
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

# ==========================================
# 1. Basic Blocks & Encoders
# ==========================================

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

class ConditionEncoder(nn.Module):
    """
    Encodes the LR input (or Reference) into Latent Space.
    Downsamples by factor of 4 (Scale s=4).
    """
    def __init__(self, in_channels=1, latent_dim=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
        )
        # Downsample x2
        self.layer1 = ResBlock(32, 48, stride=2)
        # Downsample x2 (Total x4)
        self.layer2 = ResBlock(48, latent_dim, stride=2)
        self.layer3 = ResBlock(latent_dim, latent_dim, stride=1)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# ==========================================
# 2. Diffusion Transformer (DiT) Components
# ==========================================

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)

class DiTBlock(nn.Module):
    """
    Standard Transformer Block with AdaLN (Adaptive Layer Norm) for timestep 
    and Cross-Attention for Condition.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Cross Attention: Query from x, Key/Value from Condition
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        
        # AdaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, t, condition):
        # x: (B, L, C), t: (B, C), condition: (B, L_cond, C)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)
        
        # Self Attention
        x_norm = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # Cross Attention (Conditioning)
        # Assuming condition is the Reference Latents
        x_norm2 = self.norm2(x)
        cross_out, _ = self.cross_attn(x_norm2, condition, condition)
        x = x + cross_out 
        
        # MLP
        x_norm3 = self.norm3(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x_norm3)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x

class ConditionalDiT(nn.Module):
    def __init__(self, input_size=16, patch_size=2, in_channels=64, hidden_size=64, depth=6, num_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.x_embedder = nn.Linear(in_channels * patch_size * patch_size, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.cond_embedder = nn.Linear(in_channels * patch_size * patch_size, hidden_size)
        
        # Positional Embedding
        num_patches = (input_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, patch_size * patch_size * in_channels, bias=True)
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize pos_embed with sin-cos 
        pass 

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * C)
        """
        c = self.in_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], c, h * p, h * p)
        return imgs

    def patchify(self, x):
        """
        x: (N, C, H, W)
        """
        n, c, h, w = x.shape
        p = self.patch_size
        x = x.reshape(n, c, h // p, p, w // p, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(n, (h // p) * (w // p), p * p * c)
        return x

    def forward(self, x, t, condition_latents):
        """
        x: Noisy Latent (B, C, H, W)
        t: Timestep (B,)
        condition_latents: Reference/LR Latent (B, C, H, W)
        """
        # Patchify Inputs
        x = self.patchify(x) # (B, L, D)
        x = self.x_embedder(x) + self.pos_embed
        
        condition = self.patchify(condition_latents)
        condition = self.cond_embedder(condition)
        
        t = self.t_embedder(t)
        
        for block in self.blocks:
            x = block(x, t, condition)
            
        x = self.final_layer(x)
        x = self.unpatchify(x)
        return x

# ==========================================
# 3. Hybrid Mamba Decoder
# ==========================================

class ContentPreservationModule(nn.Module):
    """
    Fuses Global Topology from LR (Target) into the Clean Latent.
    """
    def __init__(self, dim):
        super().__init__()
        self.norm_clean = nn.LayerNorm(dim)
        self.norm_tar = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, clean_latent, target_feat):
        # Reshape to sequence (B, C, H, W) -> (B, L, C)
        b, c, h, w = clean_latent.shape
        clean_seq = clean_latent.permute(0, 2, 3, 1).flatten(1, 2)
        tar_seq = target_feat.permute(0, 2, 3, 1).flatten(1, 2)

        q = self.norm_clean(clean_seq)
        k = v = self.norm_tar(tar_seq)

        out, _ = self.cross_attn(q, k, v)
        out = self.proj(out)
        
        # Add residual to clean latent
        out = out + clean_seq
        return out.view(b, h, w, c).permute(0, 3, 1, 2)

class MambaVisionMixer(nn.Module):
    """
    Paper: "Combines Linear and Conv1D projections with a state-space core".
    If mamba_ssm is installed, uses it. Else, approximates with Conv1D + Gating.
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        
        if HAS_MAMBA:
            self.mixer = Mamba(
                d_model=dim, 
                d_state=16,  
                d_conv=4,    
                expand=2,    
            )
        else:
            # Fallback approximation for portability
            self.proj_in = nn.Linear(dim, dim * 2)
            self.conv = nn.Conv1d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2)
            self.act = nn.SiLU()
            self.proj_out = nn.Linear(dim, dim)
            
    def forward(self, x):
        # x: (B, L, C)
        if HAS_MAMBA:
            out = self.mixer(self.norm(x))
            return out + x
        else:
            residual = x
            x = self.norm(x)
            x = self.proj_in(x)
            x = x.permute(0, 2, 1) # B, 2C, L
            x = self.conv(x)
            x = self.act(x)
            x = x.permute(0, 2, 1) # B, L, 2C
            x1, x2 = x.chunk(2, dim=-1)
            x = x1 * x2 # Gating mechanism roughly simulating selection
            x = self.proj_out(x)
            return x + residual

class HybridMambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mixer = MambaVisionMixer(dim)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        x_seq = x.permute(0, 2, 3, 1).flatten(1, 2) # (B, L, C)
        
        x_seq = self.mixer(x_seq)
        
        # FFN
        x_ffn = self.norm(x_seq)
        x_ffn = self.ffn(x_ffn)
        x_seq = x_seq + x_ffn
        
        return x_seq.view(b, h, w, c).permute(0, 3, 1, 2)

class HybridMambaDecoder(nn.Module):
    def __init__(self, latent_dim=64, out_channels=1, num_stages=4):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Stages to upsample 4x total (assume 2x then 2x)
        # Stage 1: Keep res
        self.cp1 = ContentPreservationModule(latent_dim)
        self.hmb1 = nn.Sequential(*[HybridMambaBlock(latent_dim) for _ in range(2)])
        
        # Stage 2: Upsample x2
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_up1 = nn.Conv2d(latent_dim, latent_dim, 3, 1, 1)
        self.cp2 = ContentPreservationModule(latent_dim)
        self.hmb2 = nn.Sequential(*[HybridMambaBlock(latent_dim) for _ in range(2)])
        
        # Stage 3: Upsample x2 (Total x4)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_up2 = nn.Conv2d(latent_dim, latent_dim, 3, 1, 1)
        self.cp3 = ContentPreservationModule(latent_dim)
        self.hmb3 = nn.Sequential(*[HybridMambaBlock(latent_dim) for _ in range(2)])
        
        # Final Projection
        self.final_conv = nn.Conv2d(latent_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, clean_latent, lr_feat):
        # lr_feat is the skip connection from the LR encoder (usually top level)
        # In this simplified implementation, we use lr_feat as topological guidance
        
        # Stage 1
        x = self.cp1(clean_latent, clean_latent) # Self-ref for structure at latent level
        x = self.hmb1(x)
        
        # Stage 2 (Upsample)
        x = self.up1(x)
        x = self.conv_up1(x)
        # Resize LR feat to match current resolution for CP
        lr_resized_1 = F.interpolate(lr_feat, size=x.shape[2:], mode='bilinear')
        x = self.cp2(x, lr_resized_1)
        x = self.hmb2(x)
        
        # Stage 3 (Upsample)
        x = self.up2(x)
        x = self.conv_up2(x)
        lr_resized_2 = F.interpolate(lr_feat, size=x.shape[2:], mode='bilinear')
        x = self.cp3(x, lr_resized_2)
        x = self.hmb3(x)
        
        out = self.final_conv(x)
        return out

# ==========================================
# 4. Full DiTMSR System
# ==========================================

class DiTMSR(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.latent_dim = 64
        
        # Encoders
        self.lr_encoder = ConditionEncoder(in_channels=1, latent_dim=self.latent_dim)
        # Reference Encoder (Optional, can share weights if same modality, but 64mT vs 3T differ)
        self.ref_encoder = ConditionEncoder(in_channels=1, latent_dim=self.latent_dim)
        
        # Diffusion Model
        self.dit = ConditionalDiT(input_size=16, in_channels=self.latent_dim, hidden_size=128, depth=6)
        
        # Decoder
        self.decoder = HybridMambaDecoder(latent_dim=self.latent_dim, out_channels=1)

    def encode(self, x):
        return self.lr_encoder(x)

    def forward_stage1(self, lr_img, ref_img):
        """
        Stage 1: AE Training (No Diffusion).
        Encode LR -> (Latent) -> Decode -> SR.
        """
        latent = self.lr_encoder(lr_img)
        
        # In Stage 1, we simulate the 'Clean Latent' as the encoder output directly
        # The paper says: "LR and HR encoders... decoded by Hybrid Mamba Decoder"
        sr_img = self.decoder(latent, latent) 
        return sr_img

    def forward_diffusion_train(self, lr_img, ref_img, t):
        """
        Stage 2: Training the DiT.
        1. Encode LR -> Latent (x0).
        2. Encode Ref -> Condition.
        3. Add noise to x0 -> xt.
        4. DiT predicts noise.
        """
        with torch.no_grad():
            x0 = self.lr_encoder(lr_img)
            condition = self.ref_encoder(ref_img)
        
        # Add noise
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        
        # Predict noise
        pred_noise = self.dit(x_t, t, condition)
        return pred_noise, noise

    def forward_inference(self, lr_img, ref_img, steps=15):
        """
        Inference: Encode Ref -> Random Noise -> Denoise -> Decode.
        """
        b = lr_img.shape[0]
        condition = self.ref_encoder(ref_img)
        
        # Start from noise
        shape = (b, self.latent_dim, 16, 16) # Latent size (64/4 = 16)
        img = torch.randn(shape, device=self.device)
        
        # Simple DDIM-like sampling loop (Simplified)
        for i in reversed(range(0, steps)):
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            pred_noise = self.dit(img, t, condition)
            
            # Update (Simplified Euler or DDPM step)
            alpha = self.alphas_cumprod[i]
            alpha_prev = self.alphas_cumprod[i-1] if i > 0 else torch.tensor(1.0).to(self.device)
            sigma = (1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev) # Posterior variance
            
            # Remove noise
            pred_x0 = (img - (1-alpha).sqrt() * pred_noise) / alpha.sqrt()
            dir_xt = (1 - alpha_prev - sigma).sqrt() * pred_noise
            img = alpha_prev.sqrt() * pred_x0 + dir_xt
            
        # Decode
        # We need the LR feature for content preservation. 
        # In inference, we can run the LR encoder to get the structural guidance
        lr_feat = self.lr_encoder(lr_img)
        sr = self.decoder(img, lr_feat)
        return sr

    def set_noise_schedule(self, num_steps=1000):
        self.num_steps = num_steps
        beta = torch.linspace(0.0001, 0.02, num_steps).to(self.device)
        alpha = 1. - beta
        self.alphas_cumprod = torch.cumprod(alpha, dim=0)

    def q_sample(self, x0, t, noise):
        # x_t = sqrt(alpha_bar) * x0 + sqrt(1 - alpha_bar) * epsilon
        alpha_t = self.alphas_cumprod[t][:, None, None, None]
        return alpha_t.sqrt() * x0 + (1 - alpha_t).sqrt() * noise