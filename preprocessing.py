import random
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from nibabel.processing import resample_from_to

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

def load_pair_resample_normalize(lf_path, hf_path, interp_order=1):
    lf_img = nib.load(lf_path)
    hf_img = nib.load(hf_path)

    lf_resampled_img = resample_from_to(lf_img, hf_img, order=interp_order)

    lf = lf_resampled_img.get_fdata().astype(np.float32)
    hf = hf_img.get_fdata().astype(np.float32)

    lf, hf = preprocess_pair_from_lf_stats(lf, hf)

    return lf, hf  # numpy arrays, same shape (179,221,200)

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
    Returns 2.5D LF/HF patch pairs.
    Each __getitem__ picks an XY patch and a Z-center slice, then stacks k
    adjacent LF slices (channels) and predicts the center HF slice.
    """
    def __init__(self, pairs, patch_size=96, patches_per_volume=64, cache_volumes=True,
                 tissue_sampling=True, foreground_percentile=20, min_foreground_ratio=0.05, max_tries=20,
                 augment=False, stack_size=5):
        """
        pairs: list of (lf_path, hf_path)
        patches_per_volume: how many patches to draw per volume per epoch
        cache_volumes: cache preprocessed volumes in RAM to speed up epochs
        stack_size: number of adjacent slices for 2.5D input (must be odd)
        """
        if stack_size % 2 == 0:
            raise ValueError("stack_size must be odd")
        self.pairs = pairs
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.cache_volumes = cache_volumes
        self.tissue_sampling = tissue_sampling
        self.foreground_percentile = foreground_percentile
        self.min_foreground_ratio = min_foreground_ratio
        self.max_tries = max_tries
        self.augment = augment
        self.stack_size = stack_size
        self._cache = {}  # idx -> (lf_np, hf_np, mask)

        # Make dataset length = number of "patch samples" per epoch
        self._length = len(pairs) * patches_per_volume

    def __len__(self):
        return self._length

    def _get_volume_pair(self, vol_idx):
        if self.cache_volumes and vol_idx in self._cache:
            return self._cache[vol_idx]

        lf_path, hf_path = self.pairs[vol_idx]
        lf, hf = load_pair_resample_normalize(lf_path, hf_path, interp_order=1)
        mask = None
        if self.tissue_sampling:
            mask = compute_foreground_mask(lf, percentile=self.foreground_percentile)

        if self.cache_volumes:
            self._cache[vol_idx] = (lf, hf, mask)
        return lf, hf, mask

    def __getitem__(self, idx):
        vol_idx = idx // self.patches_per_volume
        lf, hf, mask = self._get_volume_pair(vol_idx)

        z = random_z_index(lf.shape, self.stack_size)

        x = y = 0
        for _ in range(self.max_tries):
            x, y = random_xy_coords(lf.shape, self.patch_size)
            if mask is None:
                break
            patch_mask = mask[x:x + self.patch_size, y:y + self.patch_size, z]
            if patch_mask.mean() >= self.min_foreground_ratio:
                break

        lf_stack = extract_slice_stack(lf, x, y, z, self.patch_size, self.stack_size)
        hf_slice = extract_xy_patch(hf, x, y, z, self.patch_size)

        if self.augment:
            lf_stack, hf_slice = _augment_pair_2d(lf_stack, hf_slice)
            lf_stack = lf_stack.copy()
            hf_slice = hf_slice.copy()

        # to torch: LF (C, H, W), HF (1, H, W)
        lf_t = torch.from_numpy(lf_stack)
        hf_t = torch.from_numpy(hf_slice)[None, ...]
        return lf_t, hf_t