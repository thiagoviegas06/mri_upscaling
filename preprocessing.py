import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from nibabel.processing import resample_from_to

# ---------- preprocessing ----------
def preprocess_volume(volume, clip_percentiles=(1, 99), eps=1e-8):
    lo, hi = np.percentile(volume, clip_percentiles)
    volume = np.clip(volume, lo, hi)
    return ((volume - lo) / (hi - lo + eps)).astype(np.float32)

def load_pair_resample_normalize(lf_path, hf_path, interp_order=1):
    lf_img = nib.load(lf_path)
    hf_img = nib.load(hf_path)

    lf_resampled_img = resample_from_to(lf_img, hf_img, order=interp_order)

    lf = lf_resampled_img.get_fdata().astype(np.float32)
    hf = hf_img.get_fdata().astype(np.float32)

    lf = preprocess_volume(lf)
    hf = preprocess_volume(hf)

    return lf, hf  # numpy arrays, same shape (179,221,200)

def random_patch_coords(vol_shape, patch_size):
    # vol_shape: (X,Y,Z)
    x_max = vol_shape[0] - patch_size
    y_max = vol_shape[1] - patch_size
    z_max = vol_shape[2] - patch_size
    if min(x_max, y_max, z_max) < 0:
        raise ValueError(f"Patch size {patch_size} too large for volume shape {vol_shape}")
    x = random.randint(0, x_max)
    y = random.randint(0, y_max)
    z = random.randint(0, z_max)
    return x, y, z

def extract_patch(vol, x, y, z, patch_size):
    return vol[x:x+patch_size, y:y+patch_size, z:z+patch_size]

# ---------- dataset ----------
class MRIPatchDataset(Dataset):
    """
    Returns random LF/HF patch pairs.
    Each __getitem__ picks a random patch from one subject volume.
    """
    def __init__(self, pairs, patch_size=96, patches_per_volume=64, cache_volumes=True):
        """
        pairs: list of (lf_path, hf_path)
        patches_per_volume: how many patches to draw per volume per epoch
        cache_volumes: cache preprocessed volumes in RAM to speed up epochs
        """
        self.pairs = pairs
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.cache_volumes = cache_volumes
        self._cache = {}  # idx -> (lf_np, hf_np)

        # Make dataset length = number of "patch samples" per epoch
        self._length = len(pairs) * patches_per_volume

    def __len__(self):
        return self._length

    def _get_volume_pair(self, vol_idx):
        if self.cache_volumes and vol_idx in self._cache:
            return self._cache[vol_idx]

        lf_path, hf_path = self.pairs[vol_idx]
        lf, hf = load_pair_resample_normalize(lf_path, hf_path, interp_order=1)

        if self.cache_volumes:
            self._cache[vol_idx] = (lf, hf)
        return lf, hf

    def __getitem__(self, idx):
        vol_idx = idx // self.patches_per_volume
        lf, hf = self._get_volume_pair(vol_idx)

        x, y, z = random_patch_coords(lf.shape, self.patch_size)
        lf_p = extract_patch(lf, x, y, z, self.patch_size)
        hf_p = extract_patch(hf, x, y, z, self.patch_size)

        # to torch: (C, X, Y, Z)
        lf_t = torch.from_numpy(lf_p)[None, ...]
        hf_t = torch.from_numpy(hf_p)[None, ...]
        return lf_t, hf_t