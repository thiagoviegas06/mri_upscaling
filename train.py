import contextlib
import torch
from torch.amp import autocast
import torch.nn.functional as F

def _gaussian_kernel_1d(window_size, sigma, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()

def _gaussian_kernel_3d(window_size, sigma, channels, device, dtype):
    k1d = _gaussian_kernel_1d(window_size, sigma, device, dtype)
    k3d = k1d[:, None, None] * k1d[None, :, None] * k1d[None, None, :]
    k3d = k3d / k3d.sum()
    kernel = k3d[None, None, ...].repeat(channels, 1, 1, 1, 1)
    return kernel

def ssim_3d(x, y, window_size=7, sigma=1.5, data_range=1.0):
    # x, y: (N, C, D, H, W)
    channels = x.size(1)
    kernel = _gaussian_kernel_3d(window_size, sigma, channels, x.device, x.dtype)
    padding = window_size // 2

    mu_x = F.conv3d(x, kernel, padding=padding, groups=channels)
    mu_y = F.conv3d(y, kernel, padding=padding, groups=channels)

    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv3d(x * x, kernel, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv3d(y * y, kernel, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv3d(x * y, kernel, padding=padding, groups=channels) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))
    return ssim_map.mean()

def train_one_epoch(model, loader, optim, device, scaler):
    model.train()
    running = 0.0

    for lf, hf in loader:
        lf = lf.to(device, non_blocking=True)
        hf = hf.to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)

        amp_ctx = autocast(device_type="cuda") if device == "cuda" else contextlib.nullcontext()
        with amp_ctx:
            pred = model(lf)
            l1 = F.l1_loss(pred, hf)
            ssim = ssim_3d(pred, hf, data_range=1.0)
            loss = l1 + (1.0 - ssim)

        if scaler is not None and device == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()

        running += loss.item()

    return running / max(1, len(loader))

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    running = 0.0

    for lf, hf in loader:
        lf = lf.to(device, non_blocking=True)
        hf = hf.to(device, non_blocking=True)

        amp_ctx = autocast(device_type="cuda") if device == "cuda" else contextlib.nullcontext()
        with amp_ctx:
            pred = model(lf)
            l1 = F.l1_loss(pred, hf)
            ssim = ssim_3d(pred, hf, data_range=1.0)
            loss = l1 + (1.0 - ssim)

        running += loss.item()

    return running / max(1, len(loader))