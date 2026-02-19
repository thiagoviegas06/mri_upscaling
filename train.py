# train.py (patched + trimmed)
import contextlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from preprocessing import load_pair_resample_normalize
from mri_resolution.metric import _normalize_01, _gaussian_kernel_2d, _ssim_components, compute_ms_ssim


# ---------------------------
# EMA
# ---------------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
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


# ---------------------------
# MS-SSIM (2D) for training/validation loss
# ---------------------------
def _gaussian_kernel_2d(window_size, sigma, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]
    return kernel / kernel.sum()

def _ssim_components_2d(x, y, kernel, c1, c2):
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
    """
    x,y: (N,C,H,W) in [0,1]. We assume caller clamps; this function is differentiable.
    """
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

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


def compute_loss(pred, target, ms_weight=0.7):
    # Pred is already [0,1] from the model's Sigmoid
    # Target is already [0,1] from your loader
    
    # We still clamp just to be safe from floating point errors
    p = pred.clamp(0.0, 1.0)
    t = target.clamp(0.0, 1.0)

    l1 = F.l1_loss(p, t)
    ms_ssim_val = ms_ssim_2d_torch(p, t)
    
    return (1.0 - ms_weight) * l1 + ms_weight * (1.0 - ms_ssim_val)


# ---------------------------
# Train / Validate
# ---------------------------
def _to_2d_slices(pred, hf):
    """
    pred,hf: (N,C,H,W) or (N,C,D,H,W)
    Returns pred_2d, hf_2d as (N*D, C, H, W) if 5D else unchanged.
    """
    if pred.dim() == 5:
        N, C, D, H, W = pred.shape
        pred_2d = pred.permute(0, 2, 1, 3, 4).reshape(N * D, C, H, W)
        hf_2d = hf.permute(0, 2, 1, 3, 4).reshape(N * D, C, H, W)
        return pred_2d, hf_2d
    return pred, hf


def train_one_epoch(model, loader, optim, device, scaler=None, ema=None,
                    ms_weight=0.9):
    model.train()
    running = 0.0

    for lf, hf in loader:
        lf = lf.to(device, non_blocking=True)
        hf = hf.to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)

        amp_ctx = autocast(device_type="cuda") if device == "cuda" else contextlib.nullcontext()
        with amp_ctx:
            pred = model(lf)
            pred_2d, hf_2d = _to_2d_slices(pred, hf)
            loss = compute_loss(pred_2d, hf_2d, ms_weight=ms_weight)

        if scaler is not None and device == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()

        if ema is not None:
            ema.update(model)

        running += float(loss.item())

    return running / max(1, len(loader))


@torch.no_grad()
def validate(model, loader, device, ms_weight=0.9):
    model.eval()
    running = 0.0

    for lf, hf in loader:
        lf = lf.to(device, non_blocking=True)
        hf = hf.to(device, non_blocking=True)

        amp_ctx = autocast(device_type="cuda") if device == "cuda" else contextlib.nullcontext()
        with amp_ctx:
            pred = model(lf)
            pred_2d, hf_2d = _to_2d_slices(pred, hf)
            loss = compute_loss(pred_2d, hf_2d, ms_weight=ms_weight)

        running += float(loss.item())

    return running / max(1, len(loader))


# ---------------------------
# Optional: volume-level metric validation (kept because you had it)
# If you truly don’t use it anymore, delete this entire section.
# ---------------------------
_validation_volume_cache = {}

def _gaussian_window_2d(patch_size, sigma=None):
    if sigma is None:
        sigma = patch_size / 5.0
    coords = np.arange(patch_size) - (patch_size - 1) / 2.0
    g1d = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    g2d = g1d[:, None] * g1d[None, :]
    return g2d.astype(np.float32)

def _start_indices(dim, patch_size, stride):
    if dim <= patch_size:
        return [0]
    idxs = list(range(0, dim - patch_size + 1, stride))
    if idxs[-1] != dim - patch_size:
        idxs.append(dim - patch_size)
    return idxs

def _slice_stack(volume, z_center, stack_size):
    half = stack_size // 2
    depth = volume.shape[2]
    slices = []
    for z in range(z_center - half, z_center + half + 1):
        zc = min(max(z, 0), depth - 1)
        slices.append(volume[:, :, zc])
    return np.stack(slices, axis=0)

@torch.inference_mode()
def predict_volume_batched_xy(
    stage1, volume, refiner=None,
    patch_size=96, stride=48, device="cpu", stack_size=7,
    use_amp=True,
    microbatch=32,
):
    stage1.eval()
    if refiner is not None:
        refiner.eval()

    x_starts = _start_indices(volume.shape[0], patch_size, stride)
    y_starts = _start_indices(volume.shape[1], patch_size, stride)
    depth = volume.shape[2]

    pred_vol = np.zeros_like(volume, dtype=np.float32)
    gaussian_window = _gaussian_window_2d(patch_size).astype(np.float32)

    if use_amp and (device != "cpu"):
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        autocast_ctx = contextlib.nullcontext()

    for z in range(depth):
        accum = np.zeros((volume.shape[0], volume.shape[1]), dtype=np.float32)
        weight = np.zeros((volume.shape[0], volume.shape[1]), dtype=np.float32)

        stack_full = _slice_stack(volume, z, stack_size)  # (C, X, Y)

        patches = []
        coords = []
        for x in x_starts:
            for y in y_starts:
                patches.append(stack_full[:, x:x + patch_size, y:y + patch_size])
                coords.append((x, y))

        patches_np = np.stack(patches, axis=0).astype(np.float32)  # (B,C,H,W)
        patch_t = torch.from_numpy(patches_np).to(device, non_blocking=True)

        preds_all = []
        with autocast_ctx:
            for s in range(0, patch_t.shape[0], microbatch):
                mb = patch_t[s:s + microbatch]
                if refiner is None:
                    out = stage1(mb)  # (mb,1,H,W)
                else:
                    y1 = stage1(mb)
                    inp = torch.cat([mb, y1], dim=1)
                    delta = refiner(inp)
                    limit = 0.2
                    delta = limit * torch.tanh(delta / limit)
                    out = y1 + delta
                preds_all.append(out)

        pred_t = torch.cat(preds_all, dim=0)  # (B,1,H,W)
        preds = pred_t.squeeze(1).float().cpu().numpy()  # (B,H,W)

        for i, (x, y) in enumerate(coords):
            accum[x:x + patch_size, y:y + patch_size] += preds[i] * gaussian_window
            weight[x:x + patch_size, y:y + patch_size] += gaussian_window

        pred_vol[:, :, z] = accum / np.maximum(weight, 1e-8)

    return pred_vol


@torch.no_grad()
def validate_metric(stage1, pairs, device, ema=None, refiner=None, patch_size=96,
                    stride=48, max_volumes=None, slice_stride=1, stack_size=7):
    """
    Kaggle-matching validation:
    - full-volume stitched prediction
    - per-slice min-max normalization to [0,1] (exactly like Kaggle _normalize_01)
    - Kaggle MS-SSIM using fftconvolve(mode='valid') + even-dim trimming downsampling
    Returns: (mean_ms_ssim, total_slices)
    """
    stage1.eval()
    if refiner is not None:
        refiner.eval()

    total_ms_ssim = 0.0
    total_slices = 0

    for idx, (lf_path, hf_path) in enumerate(pairs):
        if max_volumes is not None and idx >= max_volumes:
            break

        cache_key = (lf_path, hf_path)
        if cache_key in _validation_volume_cache:
            lf, hf = _validation_volume_cache[cache_key]
        else:
            lf, hf, _ = load_pair_resample_normalize(lf_path, hf_path, interp_order=1, normalize=False)
            _validation_volume_cache[cache_key] = (lf, hf)

        ema_backup = None
        if ema is not None:
            ema_backup = ema.apply_to(stage1)

        pred = predict_volume_batched_xy(
            stage1, lf, refiner=refiner,
            patch_size=patch_size, stride=stride,
            device=device, stack_size=stack_size,
            use_amp=(device == "cuda"),
        )

        if ema is not None:
            ema.restore(stage1, ema_backup)

        # IMPORTANT: Kaggle does NOT clip; it min-max normalizes each slice.
        # Clipping can still be okay, but to match Kaggle most closely, skip it here.
        pred = np.clip(pred, 0.0, 1.0)
        
        for z in range(0, hf.shape[2], slice_stride):
            gt_slice = hf[:, :, z]
            pr_slice = pred[:, :, z]

            gt_norm = _normalize_01(gt_slice)
            pr_norm = _normalize_01(pr_slice)

            total_ms_ssim += compute_ms_ssim(gt_norm, pr_norm)
            total_slices += 1

    if total_slices == 0:
        return 0.0, 0

    return total_ms_ssim / total_slices, total_slices
