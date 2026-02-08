import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from nibabel.processing import resample_from_to

# ---------- preprocessing ----------
def preprocess_volume(volume, clip_percentiles=(1, 99), eps=1e-8):
    # Clip extreme intensities and scale to [0, 1] for stable training.
    lo, hi = np.percentile(volume, clip_percentiles)
    volume = np.clip(volume, lo, hi)
    return ((volume - lo) / (hi - lo + eps)).astype(np.float32)

def load_pair_resample_normalize(lf_path, hf_path, interp_order=1):
    # Load LF/HF volumes, resample LF to HF grid, and normalize both.
    lf_img = nib.load(lf_path)
    hf_img = nib.load(hf_path)

    lf_resampled_img = resample_from_to(lf_img, hf_img, order=interp_order)

    lf = lf_resampled_img.get_fdata().astype(np.float32)
    hf = hf_img.get_fdata().astype(np.float32)

    lf = preprocess_volume(lf)
    hf = preprocess_volume(hf)

    return lf, hf  # numpy arrays, same shape (179,221,200)

def random_patch_coords(vol_shape, patch_size, mask=None, min_foreground_ratio=0.05, max_tries=20):
    # Randomly sample patch coordinates, optionally biasing toward tissue via a mask.
    # vol_shape: (X,Y,Z)
    x_max = vol_shape[0] - patch_size
    y_max = vol_shape[1] - patch_size
    z_max = vol_shape[2] - patch_size
    if min(x_max, y_max, z_max) < 0:
        raise ValueError(f"Patch size {patch_size} too large for volume shape {vol_shape}")
    x = y = z = 0
    for _ in range(max_tries):
        x = random.randint(0, x_max)
        y = random.randint(0, y_max)
        z = random.randint(0, z_max)
        if mask is None:
            return x, y, z
        patch_mask = mask[x:x+patch_size, y:y+patch_size, z:z+patch_size]
        if patch_mask.mean() >= min_foreground_ratio:
            return x, y, z
    return x, y, z

def compute_foreground_mask(volume, percentile=20):
    # Build a simple tissue mask using a percentile intensity threshold.
    thresh = np.percentile(volume, percentile)
    return volume > thresh

def extract_patch(vol, x, y, z, patch_size):
    # Extract a cubic patch from a 3D volume.
    return vol[x:x+patch_size, y:y+patch_size, z:z+patch_size]

def apply_augmentations(lf_patch, hf_patch, flip_prob=0.5, noise_std=0.01, intensity_jitter=0.05):
    # Apply simple 3D augmentations consistently to LF/HF patches.
    # Random flips
    for axis in (0, 1, 2):
        if random.random() < flip_prob:
            lf_patch = np.flip(lf_patch, axis=axis)
            hf_patch = np.flip(hf_patch, axis=axis)

    # Intensity jitter (same scale for LF/HF to preserve correspondence)
    scale = 1.0 + random.uniform(-intensity_jitter, intensity_jitter)
    lf_patch = np.clip(lf_patch * scale, 0.0, 1.0)
    hf_patch = np.clip(hf_patch * scale, 0.0, 1.0)

    # Add small Gaussian noise to LF only
    if noise_std > 0:
        lf_patch = np.clip(lf_patch + np.random.normal(0.0, noise_std, size=lf_patch.shape), 0.0, 1.0)

    return lf_patch, hf_patch

# ---------- dataset ----------
class MRIPatchDataset(Dataset):
    """
    Returns random LF/HF patch pairs.
    Each __getitem__ picks a random patch from one subject volume.
    """
    def __init__(self, pairs, patch_size=96, patches_per_volume=64, cache_volumes=True,
                 tissue_sampling=True, foreground_percentile=20, min_foreground_ratio=0.05, max_tries=20,
                 augment=True, flip_prob=0.5, noise_std=0.01, intensity_jitter=0.05):
        """
        pairs: list of (lf_path, hf_path)
        patches_per_volume: how many patches to draw per volume per epoch
        cache_volumes: cache preprocessed volumes in RAM to speed up epochs
        """
        self.pairs = pairs
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.cache_volumes = cache_volumes
        self.tissue_sampling = tissue_sampling
        self.foreground_percentile = foreground_percentile
        self.min_foreground_ratio = min_foreground_ratio
        self.max_tries = max_tries
        self.augment = augment
        self.flip_prob = flip_prob
        self.noise_std = noise_std
        self.intensity_jitter = intensity_jitter
        self._cache = {}  # idx -> (lf_np, hf_np)

        # Make dataset length = number of "patch samples" per epoch
        self._length = len(pairs) * patches_per_volume

    def __len__(self):
        return self._length

    def _get_volume_pair(self, vol_idx):
        # Load (and cache) the LF/HF volume pair and optional tissue mask.
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
        # Sample a tissue-biased patch and return LF/HF tensors.
        vol_idx = idx // self.patches_per_volume
        lf, hf, mask = self._get_volume_pair(vol_idx)

        x, y, z = random_patch_coords(
            lf.shape,
            self.patch_size,
            mask=mask,
            min_foreground_ratio=self.min_foreground_ratio,
            max_tries=self.max_tries,
        )
        lf_p = extract_patch(lf, x, y, z, self.patch_size)
        hf_p = extract_patch(hf, x, y, z, self.patch_size)

        if self.augment:
            lf_p, hf_p = apply_augmentations(
                lf_p,
                hf_p,
                flip_prob=self.flip_prob,
                noise_std=self.noise_std,
                intensity_jitter=self.intensity_jitter,
            )

        # to torch: (C, X, Y, Z)
        lf_t = torch.from_numpy(lf_p)[None, ...]
        hf_t = torch.from_numpy(hf_p)[None, ...]
        return lf_t, hf_t