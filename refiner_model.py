import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class SMPRefiner(nn.Module):
    """
    Wraps a pre-trained MobileNetV2 UNet for 1-channel MRI refinement.
    """
    def __init__(self, encoder_name="mobilenet_v2", encoder_weights="imagenet"):
        super().__init__()
        
        # Instantiate a lightweight, pre-trained UNet
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=1,
            classes=1,
            activation=None
        )

    def forward(self, x):
        # Global residual learning: the model learns the high-frequency 
        # details to add to the already-upsampled input slice.
        out = self.unet(x)
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
        feat = self.unet(x) 
        B, C, D, H, W = feat.shape
        feat_2d = feat.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        refined_2d = self.refiner(feat_2d)
        out = refined_2d.reshape(B, D, C, H, W).permute(0, 2, 1, 3, 4)
        return out