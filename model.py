import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
    def forward(self, x): return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_ch)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_ch)
        self.act2 = nn.LeakyReLU(0.1, inplace=True)

        self.skip = None
        if in_ch != out_ch:
            self.skip = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        identity = x if self.skip is None else self.skip(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.act2(out + identity)
        return out

class UNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=16, dropout_p=0.1): # Changed base to 16
        super().__init__()
        self.enc1 = ResidualBlock(in_ch, base)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ResidualBlock(base, base*2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ResidualBlock(base*2, base*4)
        self.pool3 = nn.MaxPool3d(2)

        self.bott = ResidualBlock(base*4, base*8)
        self.bott_dropout = nn.Dropout3d(p=dropout_p)

        self.up3 = nn.ConvTranspose3d(base*8, base*4, 2, stride=2)
        self.dec3 = ResidualBlock(base*8, base*4) # Cat e3(4) + d3(4) = 8 channels in
        
        self.up2 = nn.ConvTranspose3d(base*4, base*2, 2, stride=2)
        self.dec2 = ResidualBlock(base*4, base*2) # Cat e2(2) + d2(2) = 4 channels in
        
        self.up1 = nn.ConvTranspose3d(base*2, base, 2, stride=2)
        self.dec1 = ResidualBlock(base*2, base)   # Cat e1(1) + d1(1) = 2 channels in

        self.out = nn.Conv3d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b  = self.bott(self.pool3(e3))
        b  = self.bott_dropout(b)

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # Global Residual Learning: Predict the missing details and add to input
        residual = self.out(d1)
        return x + residual