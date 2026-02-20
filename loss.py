import torch
import torch.nn as nn
import torch.nn.functional as F

def gaussian_1d(window_size, sigma):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()

def gaussian_2d(window_size, sigma):
    g1 = gaussian_1d(window_size, sigma)
    g2 = g1[:, None] * g1[None, :]
    return g2

class MSSSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, data_range=1.0, channels=1, alpha=0.84):
        """
        Calculates the Mixed MS-SSIM + L1 Loss.
        alpha: Weight for the MS-SSIM component. (1 - alpha) is used for L1.
        """
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range
        self.alpha = alpha
        
        # MS-SSIM Constants
        self.C1 = (0.01 * data_range) ** 2
        self.C2 = (0.03 * data_range) ** 2
        
        # Wang et al. 2003 MS-SSIM scale weights
        self.weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        
        # Create 2D Gaussian Kernel
        kernel = gaussian_2d(window_size, sigma).unsqueeze(0).unsqueeze(0)
        self.register_buffer('kernel', kernel.repeat(channels, 1, 1, 1))
        self.l1_loss = nn.L1Loss()

    def ssim_cs(self, img1, img2):
        # Convolution with gaussian kernel
        pad = self.window_size // 2
        mu1 = F.conv2d(img1, self.kernel, padding=pad, groups=img1.shape[1])
        mu2 = F.conv2d(img2, self.kernel, padding=pad, groups=img2.shape[1])
        
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, self.kernel, padding=pad, groups=img1.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.kernel, padding=pad, groups=img2.shape[1]) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.kernel, padding=pad, groups=img1.shape[1]) - mu1_mu2
        
        # Luminance (l) and Contrast/Structure (cs)
        l = (2 * mu1_mu2 + self.C1) / (mu1_sq + mu2_sq + self.C1)
        cs = (2 * sigma12 + self.C2) / (sigma1_sq + sigma2_sq + self.C2)
        return l, cs
        
    def forward(self, pred, target):
        # 1. Handle 3D Inputs: Convert (B, 1, D, H, W) to (B * D, 1, H, W)
        # This guarantees we score slice-by-slice, exactly matching the evaluation metric.
        if pred.dim() == 5:
            B, C, D, H, W = pred.shape
            pred_2d = pred.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
            target_2d = target.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        else:
            pred_2d, target_2d = pred, target
            
        self.weights = self.weights.to(pred_2d.device)
        levels = len(self.weights)
        mcs = []
        
        img1, img2 = pred_2d, target_2d
        
        for i in range(levels):
            l, cs = self.ssim_cs(img1, img2)
            
            # Spatial Mean per slice
            l_mean = l.mean(dim=(1, 2, 3))
            cs_mean = cs.mean(dim=(1, 2, 3))
            
            if i == levels - 1:
                mcs.append(l_mean * cs_mean) # Last level multiplies L and CS
            else:
                mcs.append(cs_mean)
                # Downsample 2x for next scale
                img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
                img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
        
        # Stack scales: (levels, batch)
        mcs = torch.stack(mcs, dim=0)
        mcs = torch.relu(mcs) # Ensure no negatives before power
        
        # Apply Wang weights: Π (metric_j ^ weight_j)
        msssim = torch.prod(mcs ** self.weights.view(-1, 1), dim=0).mean()
        
        # Calculate Mixed Loss
        l1 = self.l1_loss(pred, target)
        loss = self.alpha * (1 - msssim) + (1 - self.alpha) * l1
        
        return loss