import torch
import torch.nn as nn

class Discriminator3D(nn.Module):
    def __init__(self, in_ch=1, base=64):
        super().__init__()
        # A 3D PatchGAN Discriminator
        # It reduces the input volume dimensions by a factor of 8 (3 strided layers)
        
        def block(in_f, out_f, normalize=True):
            layers = [nn.Conv3d(in_f, out_f, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm3d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # Input: (B, 1, 96, 96, 96) -> Output: (B, 64, 48, 48, 48)
            *block(in_ch, base, normalize=False),
            
            # (B, 64, 48, 48, 48) -> (B, 128, 24, 24, 24)
            *block(base, base * 2),
            
            # (B, 128, 24, 24, 24) -> (B, 256, 12, 12, 12)
            *block(base * 2, base * 4),
            
            # Final output layer: Reduces to 1 channel prediction map
            # (B, 256, 12, 12, 12) -> (B, 1, 12, 12, 12)
            nn.Conv3d(base * 4, 1, kernel_size=3, padding=1) 
        )

    def forward(self, x):
        # Output is raw logits. We use BCEWithLogitsLoss or MSELoss (LSGAN) in the loop.
        return self.model(x)