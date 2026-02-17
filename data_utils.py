import os
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def make_pairs(lf_dir, hf_dir):
    pairs = []
    for fname in sorted(os.listdir(lf_dir)):
        if not (fname.endswith(".nii") or fname.endswith(".nii.gz")):
            continue
        lf_path = os.path.join(lf_dir, fname)
        hf_name = fname.replace("lowfield", "highfield")
        hf_path = os.path.join(hf_dir, hf_name)
        if os.path.exists(hf_path):
            pairs.append((lf_path, hf_path))
    return pairs

def split_pairs(pairs, val_frac=0.2, seed=42):
    pairs = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_frac))
    return pairs[n_val:], pairs[:n_val]