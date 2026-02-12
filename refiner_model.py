import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """Residual in Residual Dense Block"""
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nf=64, nb=6, gc=32):
        """
        in_nc: input channels (1 for MRI)
        out_nc: output channels (1 for MRI)
        nf: number of filters
        nb: number of blocks (23 is standard for full Real-ESRGAN, 6 is lighter)
        """
        super(RRDBNet, self).__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(feat))
        feat = feat + trunk
        out = self.conv_last(feat)
        # Global Residual Learning: Add input to output (restoration mode)
        return x + out
    

class CascadedModel(nn.Module):
    """
    Wraps the UNet and Refiner into a single model for 3D validation.
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
        # We treat the Depth (D) dimension as the batch for the 2D refiner
        B, C, D, H, W = feat.shape
        # Permute to (B, D, C, H, W) then reshape to (B*D, C, H, W)
        feat_2d = feat.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        
        # 3. Refine Slices
        refined_2d = self.refiner(feat_2d)
        
        # 4. Reshape back to 3D
        # (B*D, C, H, W) -> (B, D, C, H, W) -> (B, C, D, H, W)
        out = refined_2d.reshape(B, D, C, H, W).permute(0, 2, 1, 3, 4)
        
        return out