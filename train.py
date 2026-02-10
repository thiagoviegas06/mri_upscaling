import contextlib
import numpy as np
import torch
from torch.amp import autocast
import torch.nn.functional as F

from preprocessing import load_pair_resample_normalize


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        for name, param in model.state_dict().items():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[name] = param.detach().clone()

    def apply_to(self, model):
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)
        return backup

    def restore(self, model, backup):
        model.load_state_dict(backup, strict=False)

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state):
        self.shadow = {k: v.clone() for k, v in state.items()}

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

def _normalize_2d(x):
    x_min = x.min()
    x_max = x.max()
    if x_max - x_min > 0:
        return (x - x_min) / (x_max - x_min)
    return np.zeros_like(x)

def ssim_2d_metric(img1, img2):
    img1_norm = _normalize_2d(img1)
    img2_norm = _normalize_2d(img2)

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu1 = img1_norm.mean()
    mu2 = img2_norm.mean()

    sigma1_sq = ((img1_norm - mu1) ** 2).mean()
    sigma2_sq = ((img2_norm - mu2) ** 2).mean()
    sigma12 = ((img1_norm - mu1) * (img2_norm - mu2)).mean()

    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)

    return float(numerator / denominator)

def psnr_2d_metric(img1, img2):
    img1_norm = _normalize_2d(img1)
    img2_norm = _normalize_2d(img2)

    mse = ((img1_norm - img2_norm) ** 2).mean()
    if mse == 0:
        return 50.0

    psnr = 10 * np.log10(1.0 / mse)
    return float(min(max(psnr, 0), 50))

def _start_indices(dim, patch_size, stride):
    if dim <= patch_size:
        return [0]
    idxs = list(range(0, dim - patch_size + 1, stride))
    if idxs[-1] != dim - patch_size:
        idxs.append(dim - patch_size)
    return idxs

@torch.no_grad()
def predict_volume(model, volume, patch_size=96, stride=48, device="cpu"):
    x_starts = _start_indices(volume.shape[0], patch_size, stride)
    y_starts = _start_indices(volume.shape[1], patch_size, stride)
    z_starts = _start_indices(volume.shape[2], patch_size, stride)

    accum = np.zeros_like(volume, dtype=np.float32)
    weight = np.zeros_like(volume, dtype=np.float32)

    for x in x_starts:
        for y in y_starts:
            for z in z_starts:
                patch = volume[x:x + patch_size, y:y + patch_size, z:z + patch_size]
                patch_t = torch.from_numpy(patch)[None, None, ...].to(device)
                pred_t = model(patch_t)
                pred = pred_t.squeeze(0).squeeze(0).cpu().numpy()

                accum[x:x + patch_size, y:y + patch_size, z:z + patch_size] += pred
                weight[x:x + patch_size, y:y + patch_size, z:z + patch_size] += 1.0

    return accum / np.maximum(weight, 1e-8)

def train_one_epoch(model, loader, optim, device, scaler, ema=None):
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

        if ema is not None:
            ema.update(model)

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

@torch.no_grad()
def validate_metric(model, pairs, device, patch_size=96, stride=48, max_volumes=None, slice_stride=1):
    model.eval()
    total_ssim = 0.0
    total_psnr = 0.0
    total_slices = 0

    for idx, (lf_path, hf_path) in enumerate(pairs):
        if max_volumes is not None and idx >= max_volumes:
            break

        lf, hf = load_pair_resample_normalize(lf_path, hf_path, interp_order=1)
        pred = predict_volume(model, lf, patch_size=patch_size, stride=stride, device=device)
        pred = np.clip(pred, 0.0, 1.0)

        for z in range(0, hf.shape[2], slice_stride):
            gt_slice = hf[:, :, z]
            pred_slice = pred[:, :, z]

            total_ssim += ssim_2d_metric(gt_slice, pred_slice)
            total_psnr += psnr_2d_metric(gt_slice, pred_slice)
            total_slices += 1

    if total_slices == 0:
        return 0.0, 0.0, 0.0, 0

    mean_ssim = total_ssim / total_slices
    mean_psnr = total_psnr / total_slices
    score = 0.5 * mean_ssim + 0.5 * (mean_psnr / 50.0)
    return score, mean_ssim, mean_psnr, total_slices