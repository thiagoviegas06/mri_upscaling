import torch
import torch.nn as nn
from basicsr.archs.rrdbnet_arch import RRDBNet

class GrayscaleRealESRGAN_1x(nn.Module):
    def __init__(self, model_path, device="cuda"):
        super().__init__()
        
        # 1. Initialize the STANDARD 4x model (3 channels in/out) to align with pretrained weights
        base_model = RRDBNet(
            num_in_ch=3, 
            num_out_ch=3, 
            num_feat=64, 
            num_block=23, 
            num_grow_ch=32, 
            scale=4
        )
        
        # 2. Load the pre-trained state dict
        state_dict = torch.load(model_path, map_location=device)
        if 'params_ema' in state_dict:
            state_dict = state_dict['params_ema']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
            
        # 3. Handle 1-Channel Input:
        # Sum the weights of the first conv layer across the 3 RGB channels.
        # Shape goes from [64, 3, 3, 3] -> [64, 1, 3, 3]
        state_dict['conv_first.weight'] = state_dict['conv_first.weight'].sum(dim=1, keepdim=True)
        
        # Override the first layer in the base model to accept 1 channel
        base_model.conv_first = nn.Conv2d(1, 64, 3, 1, 1)
        
        # Load weights into base_model 
        # (strict=False because the state_dict has 4x upsampling layers we don't need)
        base_model.load_state_dict(state_dict, strict=False)
        
        # 4. SURGERY (1x Scale & 1-Channel Output):
        # Extract only the feature extraction trunk, discarding the 4x upsamplers
        self.conv_first = base_model.conv_first
        self.body = base_model.body           # The 23 Dense Blocks
        self.conv_body = base_model.conv_body # The trunk convolution
        
        # Create a brand new 1-channel output layer
        self.conv_last = nn.Conv2d(64, 1, 3, 1, 1)
        
    def forward(self, x):
        # Forward pass matching the 1x scale, 1-channel logic
        feat = self.conv_first(x)
        trunk = self.conv_body(self.body(feat))
        feat = feat + trunk
        
        # Output is exactly the same spatial size as input, with 1 channel
        out = self.conv_last(feat)
        
        # Global Residual Learning: Add input to output (restoration mode)
        return x + out


class CascadedModel(nn.Module):
    """
    Wraps the UNet and 1x Refiner into a single model for 3D validation.
    Input: (B, 1, D, H, W) -> UNet -> (B, 1, D, H, W) -> Slicing -> Refiner -> (B, 1, D, H, W)
    """
    def __init__(self, unet, refiner, device="cuda"):
        super().__init__()
        self.unet = unet
        self.refiner = refiner
        self.device = device

    def forward(self, x):
        # 1. UNet Prediction (3D)
        # x is (B, 1, D, H, W)
        feat = self.unet(x) 
        
        # 2. Reshape for Refiner (2D)
        B, C, D, H, W = feat.shape
        feat_2d = feat.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        
        # 3. Refine Slices (Dimensions remain exactly the same)
        refined_2d = self.refiner(feat_2d)
        
        # 4. Reshape back to 3D
        out = refined_2d.reshape(B, D, C, H, W).permute(0, 2, 1, 3, 4)
        
        return out