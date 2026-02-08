import contextlib
import torch
from torch.amp import autocast
import torch.nn.functional as F
import numpy as np

from preprocessing import load_pair_resample_normalize

def _gaussian_kernel_1d(window_size, sigma, device, dtype):
    # Create a normalized 1D Gaussian kernel.
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()

def _gaussian_kernel_3d(window_size, sigma, channels, device, dtype):
    # Create a normalized 3D Gaussian kernel for depth/height/width.
    k1d = _gaussian_kernel_1d(window_size, sigma, device, dtype)
    k3d = k1d[:, None, None] * k1d[None, :, None] * k1d[None, None, :]
    k3d = k3d / k3d.sum()
    kernel = k3d[None, None, ...].repeat(channels, 1, 1, 1, 1)
    return kernel

def ssim_3d(x, y, window_size=7, sigma=1.5, data_range=1.0):
    # Compute mean SSIM over 3D volumes.
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

def _compute_loss_parts(pred, target):
    # Compute individual loss components for logging.
    l1 = F.l1_loss(pred, target)
    l2 = F.mse_loss(pred, target)
    ssim = ssim_3d(pred, target, data_range=1.0)
    return l1, l2, ssim

def _compute_loss(pred, target, loss_weights):
    # Combine L1/L2/SSIM into a weighted total loss.
    l1, l2, ssim = _compute_loss_parts(pred, target)
    return (loss_weights["l1"] * l1) + (loss_weights["l2"] * l2) + (loss_weights["ssim"] * (1.0 - ssim))

def _start_indices(dim, patch_size, stride):
    # Generate sliding-window start indices with edge coverage.
    if dim <= patch_size:
        return [0]
    idxs = list(range(0, dim - patch_size + 1, stride))
    if idxs[-1] != dim - patch_size:
        idxs.append(dim - patch_size)
    return idxs

@torch.no_grad()
def _predict_volume(model, volume, patch_size=96, stride=48, device="cpu"):
    # Run sliding-window inference and stitch patches by averaging overlaps.
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

def train_one_epoch_gan(generator, discriminator, loader, opt_g, opt_d, device, scaler_g, scaler_d, loss_weights):
    generator.train()
    discriminator.train()
    
    running_g_loss = 0.0
    running_d_loss = 0.0
    running_l1 = 0.0
    running_l2 = 0.0
    running_ssim = 0.0
    
    # Loss functions
    adversarial_criterion = torch.nn.MSELoss() # LSGAN
    
    param_dtype = next(generator.parameters()).dtype
    use_amp = device == "cuda"

    for lf, hf in loader:
        lf = lf.to(device=device, dtype=param_dtype, non_blocking=True)
        hf = hf.to(device=device, dtype=param_dtype, non_blocking=True)

        # Ensure 5D tensors: (N, C, D, H, W)
        if lf.dim() == 4:
            lf = lf.unsqueeze(1)
        if hf.dim() == 4:
            hf = hf.unsqueeze(1)

        lf = lf.contiguous()
        hf = hf.contiguous()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        opt_d.zero_grad(set_to_none=True)
        
        # Use Automatic Mixed Precision (AMP) for speed/memory if available
        amp_ctx = autocast(device_type="cuda") if use_amp else contextlib.nullcontext()
        with amp_ctx:
            # 1. Real: D(HF) should be close to 1
            pred_real = discriminator(hf)
            loss_d_real = adversarial_criterion(pred_real, torch.ones_like(pred_real))
            
            # 2. Fake: D(G(LF)) should be close to 0
            # Detach fake_vol to stop gradients from flowing back into Generator here
            fake_vol = generator(lf)
            pred_fake = discriminator(fake_vol.detach()) 
            loss_d_fake = adversarial_criterion(pred_fake, torch.zeros_like(pred_fake))
            
            loss_d = 0.5 * (loss_d_real + loss_d_fake)

        # Backward D
        if scaler_d is not None and use_amp:
            scaler_d.scale(loss_d).backward()
            scaler_d.step(opt_d)
            scaler_d.update()
        else:
            loss_d.backward()
            opt_d.step()

        # -----------------
        #  Train Generator
        # -----------------
        opt_g.zero_grad(set_to_none=True)
        
        with amp_ctx:
            # We need to re-compute discriminator output for the fake volume, 
            # this time *without* detaching, so gradients flow to Generator.
            pred_fake_g = discriminator(fake_vol)
            
            # 1. Adversarial Loss: Generator wants D(Fake) to be close to 1 (fool D)
            loss_g_adv = adversarial_criterion(pred_fake_g, torch.ones_like(pred_fake_g))
            
            # 2. Content Loss: L1 + SSIM (Reuse your existing logic)
            # Note: We usually weight adversarial loss lower (e.g., 0.01 or 0.001) relative to content
            l1, l2, ssim = _compute_loss_parts(fake_vol, hf)
            loss_content = (loss_weights["l1"] * l1) + (loss_weights["ssim"] * (1.0 - ssim))
            
            # Combine losses
            # Example weight: 0.01 for adv, 1.0 for content
            loss_g = (0.01 * loss_g_adv) + loss_content

        # Backward G
        if scaler_g is not None and use_amp:
            scaler_g.scale(loss_g).backward()
            scaler_g.step(opt_g)
            scaler_g.update()
        else:
            loss_g.backward()
            opt_g.step()

        running_d_loss += loss_d.item()
        running_g_loss += loss_g.item()
        running_l1 += l1.item()
        running_l2 += l2.item()
        running_ssim += ssim.item()

    denom = max(1, len(loader))
    return {
        "loss": running_g_loss / denom,
        "g_loss": running_g_loss / denom,
        "d_loss": running_d_loss / denom,
        "l1": running_l1 / denom,
        "l2": running_l2 / denom,
        "ssim": running_ssim / denom,
    }

@torch.no_grad()
def validate(model, loader, device, loss_weights=None):
    # Validate on random patches and return averaged loss metrics.
    model.eval()
    running = 0.0
    l1_running = 0.0
    l2_running = 0.0
    ssim_running = 0.0

    if loss_weights is None:
        loss_weights = {"l1": 1.0, "l2": 0.0, "ssim": 1.0}

    param_dtype = next(model.parameters()).dtype

    for lf, hf in loader:
        lf = lf.to(device=device, dtype=param_dtype, non_blocking=True)
        hf = hf.to(device=device, dtype=param_dtype, non_blocking=True)

        if lf.dim() == 4:
            lf = lf.unsqueeze(1)
        if hf.dim() == 4:
            hf = hf.unsqueeze(1)

        lf = lf.contiguous()
        hf = hf.contiguous()

        amp_ctx = autocast(device_type="cuda") if device == "cuda" else contextlib.nullcontext()
        with amp_ctx:
            pred = model(lf)
            l1, l2, ssim = _compute_loss_parts(pred, hf)
            loss = (loss_weights["l1"] * l1) + (loss_weights["l2"] * l2) + (loss_weights["ssim"] * (1.0 - ssim))

        running += loss.item()
        l1_running += l1.item()
        l2_running += l2.item()
        ssim_running += ssim.item()

    denom = max(1, len(loader))
    return {
        "loss": running / denom,
        "l1": l1_running / denom,
        "l2": l2_running / denom,
        "ssim": ssim_running / denom,
    }

@torch.no_grad()
def validate_full_volume(model, pairs, device, loss_weights=None, patch_size=96, stride=48, max_volumes=None):
    # Validate on full volumes using sliding-window inference.
    model.eval()

    if loss_weights is None:
        loss_weights = {"l1": 1.0, "l2": 0.0, "ssim": 1.0}

    running = 0.0
    l1_running = 0.0
    l2_running = 0.0
    ssim_running = 0.0
    count = 0

    param_dtype = next(model.parameters()).dtype

    for lf_path, hf_path in pairs:
        lf, hf = load_pair_resample_normalize(lf_path, hf_path, interp_order=1)
        pred = _predict_volume(model, lf, patch_size=patch_size, stride=stride, device=device)

        pred_t = torch.from_numpy(pred)[None, None, ...].to(device=device, dtype=param_dtype)
        hf_t = torch.from_numpy(hf)[None, None, ...].to(device=device, dtype=param_dtype)
        l1, l2, ssim = _compute_loss_parts(pred_t, hf_t)
        loss = (loss_weights["l1"] * l1) + (loss_weights["l2"] * l2) + (loss_weights["ssim"] * (1.0 - ssim))

        running += loss.item()
        l1_running += l1.item()
        l2_running += l2.item()
        ssim_running += ssim.item()
        count += 1
        if max_volumes is not None and count >= max_volumes:
            break

    denom = max(1, count)
    return {
        "loss": running / denom,
        "l1": l1_running / denom,
        "l2": l2_running / denom,
        "ssim": ssim_running / denom,
    }