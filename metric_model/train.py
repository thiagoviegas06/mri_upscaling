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

LOSS_WEIGHTS = {
    "l1": 0.3,
    "mse": 0.5,
    "ssim": 0.6,
}

def _gaussian_kernel_1d(window_size, sigma, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()

def _gaussian_kernel_2d(window_size, sigma, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]
    return kernel / kernel.sum()

def _ssim_components_2d(x, y, kernel, c1, c2):
    # x, y: (N, C, H, W)
    channels = x.size(1)
    k = kernel[None, None, ...].repeat(channels, 1, 1, 1)
    padding = kernel.size(0) // 2

    mu_x = F.conv2d(x, k, padding=padding, groups=channels)
    mu_y = F.conv2d(y, k, padding=padding, groups=channels)

    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, k, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, k, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, k, padding=padding, groups=channels) - mu_xy

    luminance = (2 * mu_xy + c1) / (mu_x2 + mu_y2 + c1)
    cs = (2 * sigma_xy + c2) / (sigma_x2 + sigma_y2 + c2)
    return luminance, cs

def ms_ssim_2d_torch(x, y, window_size=11, sigma=1.5, weights=None):
    # x, y: (N, C, H, W)
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    x = x.clamp(0.0, 1.0)
    y = y.clamp(0.0, 1.0)

    kernel = _gaussian_kernel_2d(window_size, sigma, x.device, x.dtype)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mcs = []
    for scale in range(len(weights)):
        if x.shape[-2] < window_size or x.shape[-1] < window_size:
            break

        luminance, cs = _ssim_components_2d(x, y, kernel, c1, c2)
        if scale == len(weights) - 1:
            mcs.append((luminance.mean(), cs.mean()))
        else:
            mcs.append((None, cs.mean()))

        if scale < len(weights) - 1:
            x = F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=False)
            y = F.avg_pool2d(y, kernel_size=2, stride=2, ceil_mode=False)

    if not mcs:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    used_weights = weights[:len(mcs)]
    w_sum = sum(used_weights)
    used_weights = [w / w_sum for w in used_weights]

    ms_ssim = x.new_tensor(1.0)
    for i, (lum, cs_val) in enumerate(mcs):
        cs_clamped = cs_val.clamp(0.0, 1.0)
        if i == len(mcs) - 1 and lum is not None:
            lum_clamped = lum.clamp(0.0, 1.0)
            ms_ssim = ms_ssim * (lum_clamped ** used_weights[i]) * (cs_clamped ** used_weights[i])
        else:
            ms_ssim = ms_ssim * (cs_clamped ** used_weights[i])

    return ms_ssim

def compute_loss(pred, target, ms_weight=0.6):
    l1 = F.l1_loss(pred, target)
    ms_ssim_val = ms_ssim_2d_torch(pred, target)
    ms_weight = float(ms_weight)
    l1_weight = 1.0 - ms_weight
    return l1_weight * l1 + ms_weight * (1.0 - ms_ssim_val)

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

def ms_ssim_2d_metric(img1, img2):
    img1_norm = _normalize_2d(img1)
    img2_norm = _normalize_2d(img2)
    t1 = torch.from_numpy(img1_norm)[None, None, ...]
    t2 = torch.from_numpy(img2_norm)[None, None, ...]
    with torch.no_grad():
        return float(ms_ssim_2d_torch(t1, t2).item())

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

def _gaussian_window_2d(patch_size, sigma=None):
    if sigma is None:
        sigma = patch_size / 5.0
    coords = np.arange(patch_size) - (patch_size - 1) / 2.0
    g1d = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    g2d = g1d[:, None] * g1d[None, :]
    return g2d.astype(np.float32)

def _slice_stack(volume, z_center, stack_size):
    half = stack_size // 2
    depth = volume.shape[2]
    slices = []
    for z in range(z_center - half, z_center + half + 1):
        zc = min(max(z, 0), depth - 1)
        slices.append(volume[:, :, zc])
    return np.stack(slices, axis=0)

@torch.no_grad()
def predict_volume(stage1, volume, refiner=None, patch_size=96, stride=48, device="cpu", stack_size=5):
    x_starts = _start_indices(volume.shape[0], patch_size, stride)
    y_starts = _start_indices(volume.shape[1], patch_size, stride)
    depth = volume.shape[2]

    pred_vol = np.zeros_like(volume, dtype=np.float32)
    gaussian_window = _gaussian_window_2d(patch_size)

    for z in range(depth):
        accum = np.zeros((volume.shape[0], volume.shape[1]), dtype=np.float32)
        weight = np.zeros((volume.shape[0], volume.shape[1]), dtype=np.float32)
        stack_full = _slice_stack(volume, z, stack_size)

        for x in x_starts:
            for y in y_starts:
                patch = stack_full[:, x:x + patch_size, y:y + patch_size]
                patch_t = torch.from_numpy(patch)[None, ...].to(device)
                if refiner is None:
                    pred_t = stage1(patch_t)
                else:
                    y1 = stage1(patch_t)
                    inp = torch.cat([patch_t, y1], dim=1)
                    delta = refiner(inp)

                    limit = 0.2  # keep same as training; also try 0.1
                    delta = limit * torch.tanh(delta / limit)

                    pred_t = y1 + delta

                pred = pred_t.squeeze(0).squeeze(0).cpu().numpy()
                accum[x:x + patch_size, y:y + patch_size] += pred * gaussian_window
                weight[x:x + patch_size, y:y + patch_size] += gaussian_window

        pred_vol[:, :, z] = accum / np.maximum(weight, 1e-8)

    return pred_vol

def train_one_epoch(model, loader, optim, device, scaler, ema=None, ms_weight=0.6):
    model.train()
    running = 0.0

    for lf, hf in loader:
        lf = lf.to(device, non_blocking=True)
        hf = hf.to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)

        amp_ctx = autocast(device_type="cuda") if device == "cuda" else contextlib.nullcontext()
        with amp_ctx:
            pred = model(lf)
            loss = compute_loss(pred, hf, ms_weight=ms_weight)

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

def train_one_epoch_refiner(stage1, ema_stage1, refiner, loader, optim, device, scaler=None, delta_l1_weight=0.01, ms_weight=0.6):
    stage1.eval()
    refiner.train()
    running = 0.0

    for lf, hf in loader:
        lf = lf.to(device, non_blocking=True)
        hf = hf.to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)

        with torch.no_grad():
            ema_backup = None
            if ema_stage1 is not None:
                ema_backup = ema_stage1.apply_to(stage1)
            y1 = stage1(lf)
            if ema_stage1 is not None:
                ema_stage1.restore(stage1, ema_backup)

        delta = refiner(torch.cat([lf, y1], dim=1))

        limit = 0.2  # try 0.2 first; also test 0.1
        delta = limit * torch.tanh(delta / limit)

        yhat = y1 + delta

        loss_main = compute_loss(yhat, hf, ms_weight=ms_weight)
        loss_reg = delta_l1_weight * delta.abs().mean()
        loss = loss_main + loss_reg

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
def validate(model, loader, device, ms_weight=0.6):
    model.eval()
    running = 0.0

    for lf, hf in loader:
        lf = lf.to(device, non_blocking=True)
        hf = hf.to(device, non_blocking=True)

        amp_ctx = autocast(device_type="cuda") if device == "cuda" else contextlib.nullcontext()
        with amp_ctx:
            pred = model(lf)
            loss = compute_loss(pred, hf, ms_weight=ms_weight)

        running += loss.item()

    return running / max(1, len(loader))

# Persistent cache for validation volumes to avoid resampling every epoch
_validation_volume_cache = {}

@torch.no_grad()
def validate_metric(stage1, pairs, device, ema=None, refiner=None, patch_size=96, stride=48, max_volumes=None, slice_stride=1, stack_size=5):
    stage1.eval()
    if refiner is not None:
        refiner.eval()
    total_ssim = 0.0
    total_ms_ssim = 0.0
    total_psnr = 0.0
    total_slices = 0

    for idx, (lf_path, hf_path) in enumerate(pairs):
        if max_volumes is not None and idx >= max_volumes:
            break

        cache_key = (lf_path, hf_path)
        if cache_key in _validation_volume_cache:
            lf, hf = _validation_volume_cache[cache_key]
        else:
            lf, hf = load_pair_resample_normalize(lf_path, hf_path, interp_order=1)
            _validation_volume_cache[cache_key] = (lf, hf)
        ema_backup = None
        if ema is not None:
            ema_backup = ema.apply_to(stage1)
        pred = predict_volume(stage1, lf, refiner=refiner, patch_size=patch_size, stride=stride, device=device, stack_size=stack_size)
        if ema is not None:
            ema.restore(stage1, ema_backup)
        pred = np.clip(pred, 0.0, 1.0)

        for z in range(0, hf.shape[2], slice_stride):
            gt_slice = hf[:, :, z]
            pred_slice = pred[:, :, z]

            total_ssim += ssim_2d_metric(gt_slice, pred_slice)
            total_ms_ssim += ms_ssim_2d_metric(gt_slice, pred_slice)
            total_psnr += psnr_2d_metric(gt_slice, pred_slice)
            total_slices += 1

    if total_slices == 0:
        return 0.0, 0.0, 0.0, 0.0, 0

    mean_ssim = total_ssim / total_slices
    mean_ms_ssim = total_ms_ssim / total_slices
    mean_psnr = total_psnr / total_slices
    score = 0.5 * mean_ssim + 0.5 * (mean_psnr / 50.0)
    return score, mean_ssim, mean_ms_ssim, mean_psnr, total_slices
