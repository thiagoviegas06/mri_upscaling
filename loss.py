import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

class CharbonnierLoss(nn.Module):
    """L1-like loss that is differentiable at zero, standard for PSNR maximization."""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps2)
        return torch.mean(loss)

class VGGPerceptualLoss(nn.Module):
    def __init__(self, feature_layers=[3, 8, 17, 26], use_l1=True, weights=[1.0, 1.0, 1.0, 1.0]):
        """
        Args:
            feature_layers: Indices of VGG19 layers to extract features from.
                            [3, 8, 17, 26] corresponds roughly to relu1_2, relu2_2, relu3_4, relu4_4
            use_l1: If True, adds L1 pixel loss to the perceptual loss (stabilizes training).
            weights: Weights for each extracted feature layer.
        """
        super().__init__()
        self.use_l1 = use_l1
        
        # Load VGG19 pretrained on ImageNet
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        
        # Extract the features container
        self.features = vgg.features
        
        # Freeze parameters (we don't update VGG)
        for param in self.features.parameters():
            param.requires_grad = False
            
        self.layer_indices = feature_layers
        self.layer_weights = weights
        
        # ImageNet normalization statistics
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        if use_l1:
            self.l1_loss = nn.L1Loss()

    def normalize(self, x):
        """
        Converts 1-channel [0,1] MRI to 3-channel ImageNet normalized input.
        """
        # 1. Repeat channel to fake RGB: (B, 1, H, W) -> (B, 3, H, W)
        x = x.repeat(1, 3, 1, 1)
        
        # 2. Normalize using ImageNet stats
        # (Assumes input x is roughly in [0, 1])
        return (x - self.mean) / self.std

    def forward(self, input, target):
        # Normalize inputs for VGG
        input_vgg = self.normalize(input)
        target_vgg = self.normalize(target)
        
        loss = 0.0
        x = input_vgg
        y = target_vgg
        
        # Extract features and compute L1 loss between features
        for i, layer in enumerate(self.features):
            x = layer(x)
            y = layer(y)
            
            if i in self.layer_indices:
                # Get the weight for this layer index
                idx = self.layer_indices.index(i)
                w = self.layer_weights[idx]
                
                # Add weighted feature loss
                loss += w * nn.functional.l1_loss(x, y)
                
            # Stop early if we've gone past the last layer we care about
            if i >= max(self.layer_indices):
                break
                
        # Add pixel-wise L1 loss for structural consistency
        if self.use_l1:
            loss += self.l1_loss(input, target)
            
        return loss