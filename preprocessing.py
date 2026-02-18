import random
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from nibabel.processing import resample_from_to

def lf_minmax_params(lf, p_low=1, p_high=99, eps=1e-6):
    """
    Compute (lo, hi) from LF only. Robust percentile fallback to min/max.
    Returns: (lo, hi, is_degenerate)
    """
    lo, hi = np.percentile(lf, [p_low, p_high])
    if hi <= lo + eps:
        lo, hi = float(np.min(lf)), float(np.max(lf))
        if hi <= lo + eps:
            return 0.0, 1.0, True
    return float(lo), float(hi), False

def apply_lf_minmax(x, lo, hi, eps=1e-6, clip=True):
    """
    Apply min-max using provided (lo, hi).
    clip=True keeps outputs ~[0,1]. clip=False preserves out-of-range values.
    """
    if clip:
        x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + eps)
    return x.astype(np.float32)

def normalize_from_lf(lf, hf=None, *, p_low=1, p_high=99, eps=1e-6, clip=True):
    """
    Canonical normalization used by BOTH training and inference.

    - (lo, hi) computed from LF only.
    - LF always normalized.
    - If HF provided, HF normalized using SAME (lo, hi).
    """
    lo, hi, deg = lf_minmax_params(lf, p_low=p_low, p_high=p_high, eps=eps)

    if deg:
        lf_n = np.zeros_like(lf, np.float32)
        if hf is None:
            return lf_n, (lo, hi)
        return lf_n, np.zeros_like(hf, np.float32), (lo, hi)

    lf_n = apply_lf_minmax(lf, lo, hi, eps=eps, clip=clip)

    if hf is None:
        return lf_n, (lo, hi)

    hf_n = apply_lf_minmax(hf, lo, hi, eps=eps, clip=clip)
    return lf_n, hf_n, (lo, hi)


def fg_frac_np(patch2d: np.ndarray, thresh: float = 0.05) -> float:
    # patch2d: (H,W) float32
    return float((patch2d > thresh).mean())

def energy_np(patch2d: np.ndarray) -> float:
    # simple gradient/edge energy proxy
    gx = np.abs(np.diff(patch2d, axis=0)).mean()
    gy = np.abs(np.diff(patch2d, axis=1)).mean()
    return float(gx + gy)

# ---------- preprocessing ----------
def preprocess_volume(volume, clip_percentiles=(1, 99), eps=1e-8):
    lo, hi = np.percentile(volume, clip_percentiles)
    volume = np.clip(volume, lo, hi)
    return ((volume - lo) / (hi - lo + eps)).astype(np.float32)

def preprocess_pair_from_lf_stats(lf, hf, p_low=1, p_high=99, eps=1e-6):
    lo, hi = np.percentile(lf, [p_low, p_high])
    if hi <= lo + eps:
        # degenerate / near-constant volume fallback
        lo, hi = float(np.min(lf)), float(np.max(lf))
        if hi <= lo + eps:
            return np.zeros_like(lf, np.float32), np.zeros_like(hf, np.float32)

    lf_c = np.clip(lf, lo, hi)
    hf_c = np.clip(hf, lo, hi)

    lf_n = (lf_c - lo) / (hi - lo + eps)
    hf_n = (hf_c - lo) / (hi - lo + eps)
    return lf_n.astype(np.float32), hf_n.astype(np.float32)

def load_pair_resample_normalize(lf_path, hf_path, interp_order=1, *,
                                 p_low=1, p_high=99, eps=1e-6, clip=True):
    lf_img = nib.load(lf_path)
    hf_img = nib.load(hf_path)

    lf_resampled_img = resample_from_to(lf_img, hf_img, order=interp_order)

    lf = lf_resampled_img.get_fdata().astype(np.float32)
    hf = hf_img.get_fdata().astype(np.float32)

    lf, hf, (lo, hi) = normalize_from_lf(
        lf, hf, p_low=p_low, p_high=p_high, eps=eps, clip=clip
    )

    if lf.shape != hf.shape:
        raise ValueError(f"LF/HF shape mismatch: {lf.shape} vs {hf.shape}")
    if not (np.isfinite(lf).all() and np.isfinite(hf).all()):
        raise ValueError("Non-finite values found after preprocessing")

    return lf, hf, (lo, hi)

def normalize_lf_for_inference(lf, *, p_low=1, p_high=99, eps=1e-6, clip=True):
    lf_n, (lo, hi) = normalize_from_lf(
        lf, hf=None, p_low=p_low, p_high=p_high, eps=eps, clip=clip
    )
    return lf_n, (lo, hi)

def denormalize_to_lf_scale(x_n, lo, hi, eps=1e-6):
    return (x_n * (hi - lo + eps) + lo).astype(np.float32)

def normalize_lf_like_training(lf, p_low=1, p_high=99, eps=1e-6):
    """
    Normalize LF volume exactly like training does.
    Uses LF percentiles only.
    """
    lo, hi = np.percentile(lf, [p_low, p_high])

    if hi <= lo + eps:
        lo, hi = float(np.min(lf)), float(np.max(lf))
        if hi <= lo + eps:
            return np.zeros_like(lf, np.float32)

    lf_c = np.clip(lf, lo, hi)
    lf_n = (lf_c - lo) / (hi - lo + eps)

    return lf_n.astype(np.float32)



def preprocess_pair_from_lf_stats(lf, hf, p_low=1, p_high=99, eps=1e-6):
    lo, hi = np.percentile(lf, [p_low, p_high])

    if hi <= lo + eps:
        lo, hi = float(np.min(lf)), float(np.max(lf))
        if hi <= lo + eps:
            return np.zeros_like(lf, np.float32), np.zeros_like(hf, np.float32)

    lf_c = np.clip(lf, lo, hi)
    hf_c = np.clip(hf, lo, hi)

    lf_n = (lf_c - lo) / (hi - lo + eps)
    hf_n = (hf_c - lo) / (hi - lo + eps)

    return lf_n.astype(np.float32), hf_n.astype(np.float32)

def random_xy_coords(vol_shape, patch_size):
    # vol_shape: (X, Y, Z)
    x_max = vol_shape[0] - patch_size
    y_max = vol_shape[1] - patch_size
    if min(x_max, y_max) < 0:
        raise ValueError(f"Patch size {patch_size} too large for volume shape {vol_shape}")
    x = random.randint(0, x_max)
    y = random.randint(0, y_max)
    return x, y

def random_z_index(vol_shape, stack_size):
    z_dim = vol_shape[2]
    half = stack_size // 2
    if z_dim < stack_size:
        raise ValueError(f"Stack size {stack_size} too large for depth {z_dim}")
    z_min = half
    z_max = z_dim - half - 1
    return random.randint(z_min, z_max)

def compute_foreground_mask(volume, percentile=20):
    thresh = np.percentile(volume, percentile)
    return volume > thresh

def extract_xy_patch(vol, x, y, z, patch_size):
    return vol[x:x+patch_size, y:y+patch_size, z]

def extract_slice_stack(vol, x, y, z_center, patch_size, stack_size):
    half = stack_size // 2
    slices = []
    for z in range(z_center - half, z_center + half + 1):
        slices.append(extract_xy_patch(vol, x, y, z, patch_size))
    return np.stack(slices, axis=0)

def _augment_pair_2d(lf_stack, hf_slice):
    # Apply identical random flips and rotations to LF stack and HF slice.
    if random.random() < 0.5:
        lf_stack = lf_stack[:, ::-1, :]
        hf_slice = hf_slice[::-1, :]
    if random.random() < 0.5:
        lf_stack = lf_stack[:, :, ::-1]
        hf_slice = hf_slice[:, ::-1]

    k = random.randint(0, 3)
    if k:
        lf_stack = np.rot90(lf_stack, k, axes=(1, 2))
        hf_slice = np.rot90(hf_slice, k, axes=(0, 1))

    return lf_stack, hf_slice

# ---------- dataset ----------
class MRIPatchDataset(Dataset):
    """
    Minimal 2.5D LF/HF patch dataset.
    - Uses pair-consistent normalization
    - No tissue sampling
    - No augmentation
    - Random patch sampling only
    """
    def __init__(
        self,
        pairs,
        patch_size=96,
        patches_per_volume=64,
        cache_volumes=True,
        stack_size=7,
        sample_strategy="filtered",
        fg_thresh=0.10,
        energy_thresh=0.01,
        keep_bg_prob=0.30,
        max_tries=20,
        debug=False
    ):
        if stack_size % 2 == 0:
            raise ValueError("stack_size must be odd")

        self.pairs = pairs
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.cache_volumes = cache_volumes
        self.stack_size = stack_size

        self._cache = {}  # vol_idx -> (lf, hf)
        self._length = len(pairs) * patches_per_volume

        self.sample_strategy = sample_strategy
        self.fg_thresh = fg_thresh
        self.energy_thresh = energy_thresh
        self.keep_bg_prob = keep_bg_prob
        self.max_tries = max_tries
        self.debug = debug

    def __len__(self):
        return self._length

    def _get_volume_pair(self, vol_idx):
        if self.cache_volumes and vol_idx in self._cache:
            return self._cache[vol_idx]

        lf_path, hf_path = self.pairs[vol_idx]
        lf, hf = load_pair_resample_normalize(
            lf_path, hf_path, interp_order=1
        )  # <-- uses pair-consistent normalization

        if self.cache_volumes:
            self._cache[vol_idx] = (lf, hf)

        return lf, hf

    def __getitem__(self, idx):
        vol_idx = idx // self.patches_per_volume
        lf, hf = self._get_volume_pair(vol_idx)

        debug_dict = None

        # configurable constants
        fg_intensity_thresh = 0.05  # pixel intensity threshold used to compute fg fraction

        if self.sample_strategy == "filtered":
            chosen = None
            last = None
            tries = 0
            accepted_reason = None

            # ---- Decide upfront what kind of patch we want this time ----
            # keep_bg_prob now means "fraction of returned samples that are background"
            want_bg = (random.random() < self.keep_bg_prob)

            def is_bg(fg, en):
                # stricter definition of background (tunable)
                return (fg < fg_intensity_thresh) and (en < self.energy_thresh)

            def is_informative(fg, en):
                return (fg >= self.fg_thresh) and (en >= self.energy_thresh)

            target_check = is_bg if want_bg else is_informative
            target_name  = "keep_bg" if want_bg else "informative"

            for _ in range(self.max_tries):
                tries += 1
                z = random_z_index(lf.shape, self.stack_size)
                x, y = random_xy_coords(lf.shape, self.patch_size)

                lf_stack = extract_slice_stack(lf, x, y, z, self.patch_size, self.stack_size)
                hf_slice = extract_xy_patch(hf, x, y, z, self.patch_size)

                en = energy_np(hf_slice)
                fg = fg_frac_np(hf_slice, thresh=fg_intensity_thresh)

                last = (lf_stack, hf_slice, z, x, y, fg, en)

                if target_check(fg, en):
                    chosen = (lf_stack, hf_slice, z, x, y, fg, en)
                    accepted_reason = target_name
                    break

            if chosen is None:
                # fallback behavior:
                # - if we failed to find target type, return last candidate but label it
                lf_stack, hf_slice, z, x, y, fg, en = last
                accepted_reason = f"fallback_{target_name}"

            else:
                lf_stack, hf_slice, z, x, y, fg, en = chosen

        else:
            z = random_z_index(lf.shape, self.stack_size)
            x, y = random_xy_coords(lf.shape, self.patch_size)

            lf_stack = extract_slice_stack(lf, x, y, z, self.patch_size, self.stack_size)
            hf_slice = extract_xy_patch(hf, x, y, z, self.patch_size)

            fg = fg_frac_np(hf_slice, thresh=0.05)
            en = energy_np(hf_slice)
            tries = 1
            accepted_reason = "random"

        lf_t = torch.from_numpy(lf_stack).float()
        hf_t = torch.from_numpy(hf_slice)[None].float()

        if getattr(self, "debug", False):
            debug_dict = {
                "reason": accepted_reason,
                "tries": int(tries),
                "fg": float(fg),
                "en": float(en),
                "vol_idx": int(vol_idx),
                "z": int(z),
                "x": int(x),
                "y": int(y),
            }
            return lf_t, hf_t, debug_dict

        return lf_t, hf_t