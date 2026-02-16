import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
    def forward(self, x): return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 2D ops: 2.5D stacks are channels in a 2D UNet.
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(out_ch)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_ch)
        self.act2 = nn.LeakyReLU(0.1, inplace=True)

        self.skip = None
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        identity = x if self.skip is None else self.skip(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.act2(out + identity)
        return out

class AttentionGate2D(nn.Module):
    def __init__(self, in_ch, gating_ch, inter_ch):
        super().__init__()
        self.theta = nn.Conv2d(in_ch, inter_ch, 1, bias=False)
        self.phi = nn.Conv2d(gating_ch, inter_ch, 1, bias=False)
        self.psi = nn.Conv2d(inter_ch, 1, 1, bias=True)
        self.norm = nn.InstanceNorm2d(inter_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        # x: skip connection, g: gating (decoder)
        theta_x = self.theta(x)
        phi_g = self.phi(g)
        f = self.act(self.norm(theta_x + phi_g))
        attn = self.sigmoid(self.psi(f))
        return x * attn

class UNet2D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32, dropout_p=0.1):
        super().__init__()
        self.enc1 = ResidualBlock(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResidualBlock(base*2, base*4)
        self.pool3 = nn.MaxPool2d(2)

        self.bott = ResidualBlock(base*4, base*8)
        self.bott_dropout = nn.Dropout2d(p=dropout_p)

        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.att3 = AttentionGate2D(base*4, base*4, base*2)
        self.dec3 = ResidualBlock(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.att2 = AttentionGate2D(base*2, base*2, base)
        self.dec2 = ResidualBlock(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.att1 = AttentionGate2D(base, base, base//2)
        self.dec1 = ResidualBlock(base*2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b  = self.bott(self.pool3(e3))
        b  = self.bott_dropout(b)

        d3 = self.up3(b)
        e3_g = self.att3(e3, d3)
        d3 = self.dec3(torch.cat([d3, e3_g], dim=1))
        d2 = self.up2(d3)
        e2_g = self.att2(e2, d2)
        d2 = self.dec2(torch.cat([d2, e2_g], dim=1))
        d1 = self.up1(d2)
        e1_g = self.att1(e1, d1)
        d1 = self.dec1(torch.cat([d1, e1_g], dim=1))

        return self.out(d1)

class RefinerUNet2D(nn.Module):
    """
    2-level (shallower) UNet refiner.
    Input:  (B, 2, H, W)  -> concat([LF, stage1_pred])
    Output: (B, 1, H, W)  -> residual delta to add to stage1_pred
    """
    def __init__(self, in_ch=2, out_ch=1, base=24, dropout_p=0.0):
        super().__init__()
        self.enc1 = ResidualBlock(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ResidualBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck for 2-level UNet: 2*base -> 4*base
        self.bott = ResidualBlock(base * 2, base * 4)
        self.bott_dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.att2 = AttentionGate2D(base * 2, base * 2, base)
        self.dec2 = ResidualBlock(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.att1 = AttentionGate2D(base, base, base // 2)
        self.dec1 = ResidualBlock(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)              # (B, base, H, W)
        e2 = self.enc2(self.pool1(e1)) # (B, 2*base, H/2, W/2)

        # Bottleneck
        b = self.bott(self.pool2(e2))  # (B, 4*base, H/4, W/4)
        b = self.bott_dropout(b)

        # Decoder
        d2 = self.up2(b)               # (B, 2*base, H/2, W/2)
        e2_g = self.att2(e2, d2)
        d2 = self.dec2(torch.cat([d2, e2_g], dim=1))  # (B, 2*base, ...)

        d1 = self.up1(d2)              # (B, base, H, W)
        e1_g = self.att1(e1, d1)
        d1 = self.dec1(torch.cat([d1, e1_g], dim=1))  # (B, base, ...)

        return self.out(d1)            # (B, out_ch, H, W)