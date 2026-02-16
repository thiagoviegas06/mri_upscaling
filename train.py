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

# Optional: MS-SSIM for 2D slices (if available)
try:
    from pytorch_msssim import ms_ssim
    def ms_ssim_2d_metric(img1, img2):
        # img1, img2: numpy arrays, shape (H, W), values in [0, 1]
        x = torch.from_numpy(img1).float().unsqueeze(0).unsqueeze(0)
        y = torch.from_numpy(img2).float().unsqueeze(0).unsqueeze(0)
        return float(ms_ssim(x, y, data_range=1.0).item())
except ImportError:
    ms_ssim_2d_metric = None

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

def _gaussian_window_3d(patch_size, sigma=None):
    if sigma is None:
        sigma = patch_size / 5.0
    coords = np.arange(patch_size) - (patch_size - 1) / 2.0
    g1d = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    g3d = g1d[:, None, None] * g1d[None, :, None] * g1d[None, None, :]
    return g3d.astype(np.float32)

@torch.no_grad()
def predict_volume(model, volume, patch_size=96, stride=48, device="cpu"):
    x_starts = _start_indices(volume.shape[0], patch_size, stride)
    y_starts = _start_indices(volume.shape[1], patch_size, stride)
    z_starts = _start_indices(volume.shape[2], patch_size, stride)

    accum = np.zeros_like(volume, dtype=np.float32)
    weight = np.zeros_like(volume, dtype=np.float32)
    gaussian_window = _gaussian_window_3d(patch_size)

    for x in x_starts:
        for y in y_starts:
            for z in z_starts:
                patch = volume[x:x + patch_size, y:y + patch_size, z:z + patch_size]
                patch_t = torch.from_numpy(patch)[None, None, ...].to(device)
                pred_t = model(patch_t)
                pred = pred_t.squeeze(0).squeeze(0).cpu().numpy()

                accum[x:x + patch_size, y:y + patch_size, z:z + patch_size] += pred * gaussian_window
                weight[x:x + patch_size, y:y + patch_size, z:z + patch_size] += gaussian_window

    return accum / np.maximum(weight, 1e-8)

def train_one_epoch(model, loader, optim, device, scaler, ema=None, ms_ssim_weight=1.0, l1_weight=1.0):
    """
    Train one epoch using a weighted sum of L1 and (1 - MS-SSIM) losses on 2D slices.
    If MS-SSIM is unavailable, fallback to SSIM.
    """
    model.train()
    running = 0.0

    for lf, hf in loader:
        lf = lf.to(device, non_blocking=True)
        hf = hf.to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)

        amp_ctx = autocast(device_type="cuda") if device == "cuda" else contextlib.nullcontext()
        with amp_ctx:
            pred = model(lf)
            # pred, hf: (N, C, D, H, W) or (N, C, H, W)
            # For 2.5D, treat as (N, C, H, W) per slice
            if pred.dim() == 5:
                # (N, C, D, H, W) -> (N*D, C, H, W)
                N, C, D, H, W = pred.shape
                pred_2d = pred.permute(0, 2, 1, 3, 4).reshape(N*D, C, H, W)
                hf_2d = hf.permute(0, 2, 1, 3, 4).reshape(N*D, C, H, W)
            else:
                pred_2d = pred
                hf_2d = hf

            l1 = F.l1_loss(pred_2d, hf_2d)

            # Compute mean MS-SSIM (or SSIM) over all slices in batch
            ms_ssim_vals = []
            ssim_vals = []
            for i in range(pred_2d.shape[0]):
                x = pred_2d[i, 0].detach().cpu().numpy()
                y = hf_2d[i, 0].detach().cpu().numpy()
                if ms_ssim_2d_metric is not None:
                    ms_ssim_vals.append(ms_ssim_2d_metric(y, x))
                else:
                    ssim_vals.append(ssim_2d_metric(y, x))
            if ms_ssim_2d_metric is not None and ms_ssim_vals:
                ms_ssim_loss = 1.0 - float(np.mean(ms_ssim_vals))
            else:
                ms_ssim_loss = 1.0 - float(np.mean(ssim_vals))

            loss = l1_weight * l1 + ms_ssim_weight * ms_ssim_loss

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


# --- Validation slice selection state ---
_val_slice_state = {
    'epoch_group': None,
    'seed': None,
    'slice_indices': {},  # {volume_idx: [z1, z2, ...]}
}

@torch.no_grad()
def validate_metric(
    model, pairs, device, patch_size=96, stride=48, max_volumes=None, n_slices=10, epoch=None, group_size=5, use_ms_ssim=True
):
    """
    Validate using mean MS-SSIM (if available) or SSIM, on a random subset of slices per volume.
    Slices are re-sampled every `group_size` epochs, but fixed within each group for determinism.
    Args:
        model: model to evaluate
        pairs: list of (lf_path, hf_path)
        device: torch/cuda
        patch_size, stride: inference params
        max_volumes: max number of volumes to validate
        n_slices: number of slices per volume to sample
        epoch: current epoch (int)
        group_size: epochs between re-sampling slices
        use_ms_ssim: if True and available, use MS-SSIM, else SSIM
    Returns:
        mean_ms_ssim, mean_ssim, mean_psnr, total_slices
    """
    model.eval()
    total_ms_ssim = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    total_slices = 0

    # Determine epoch group and seed
    if epoch is None:
        epoch_group = 0
    else:
        epoch_group = epoch // group_size
    if _val_slice_state['epoch_group'] != epoch_group:
        # New group, re-sample
        seed = 1337 + epoch_group
        rng = np.random.RandomState(seed)
        slice_indices = {}
        for idx, (lf_path, hf_path) in enumerate(pairs):
            lf, hf = load_pair_resample_normalize(lf_path, hf_path, interp_order=1)
            nz = hf.shape[2]
            if n_slices >= nz:
                indices = list(range(nz))
            else:
                indices = sorted(rng.choice(nz, n_slices, replace=False))
            slice_indices[idx] = indices
        _val_slice_state['epoch_group'] = epoch_group
        _val_slice_state['seed'] = seed
        _val_slice_state['slice_indices'] = slice_indices
    else:
        slice_indices = _val_slice_state['slice_indices']

    for idx, (lf_path, hf_path) in enumerate(pairs):
        if max_volumes is not None and idx >= max_volumes:
            break

        lf, hf = load_pair_resample_normalize(lf_path, hf_path, interp_order=1)
        pred = predict_volume(model, lf, patch_size=patch_size, stride=stride, device=device)
        pred = np.clip(pred, 0.0, 1.0)

        indices = slice_indices.get(idx, list(range(hf.shape[2])))
        for z in indices:
            gt_slice = hf[:, :, z]
            pred_slice = pred[:, :, z]

            if use_ms_ssim and ms_ssim_2d_metric is not None:
                ms_ssim_val = ms_ssim_2d_metric(gt_slice, pred_slice)
                total_ms_ssim += ms_ssim_val
            else:
                ms_ssim_val = None
            ssim_val = ssim_2d_metric(gt_slice, pred_slice)
            psnr_val = psnr_2d_metric(gt_slice, pred_slice)
            total_ssim += ssim_val
            total_psnr += psnr_val
            total_slices += 1

    if total_slices == 0:
        return 0.0, 0.0, 0.0, 0

    mean_ms_ssim = total_ms_ssim / total_slices if (use_ms_ssim and ms_ssim_2d_metric is not None) else 0.0
    mean_ssim = total_ssim / total_slices
    mean_psnr = total_psnr / total_slices
    # For model selection, use mean_ms_ssim if available, else mean_ssim
    val_score = mean_ms_ssim if (use_ms_ssim and ms_ssim_2d_metric is not None) else mean_ssim
    # Optionally log the seed and indices for traceability
    print(f"[validate_metric] epoch_group={epoch_group}, seed={_val_slice_state['seed']}, n_slices={n_slices}")
    return val_score, mean_ms_ssim, mean_ssim, mean_psnr, total_slices