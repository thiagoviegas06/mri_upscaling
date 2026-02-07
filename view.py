import os
import matplotlib.pyplot as plt
from preprocessing import MRIPatchDataset

def make_pairs(lf_dir, hf_dir):
    """
    Build paired (LF, HF) file list based on naming convention.
    """
    pairs = []
    # Ensure directories exist
    if not os.path.exists(lf_dir) or not os.path.exists(hf_dir):
        print(f"Warning: Data directories not found: {lf_dir} or {hf_dir}")
        return []
        
    for fname in sorted(os.listdir(lf_dir)):
        if not (fname.endswith(".nii") or fname.endswith(".nii.gz")):
            continue
        lf_path = os.path.join(lf_dir, fname)
        hf_name = fname.replace("lowfield", "highfield")
        hf_path = os.path.join(hf_dir, hf_name)
        if os.path.exists(hf_path):
            pairs.append((lf_path, hf_path))
    return pairs

def visualize_middle_slice(lf_volume, hf_volume, volume_idx):
    """
    Visualizes the middle-most slice of the MRI volume pair.
    
    Args:
        lf_volume (np.array): Low field volume data (X, Y, Z)
        hf_volume (np.array): High field volume data (X, Y, Z)
        volume_idx (int): Index of the volume for display purposes
    """
    # Assuming the standard orientation (X, Y, Z), we slice along the Z-axis (depth)
    z_dim = lf_volume.shape[2]
    middle_idx = z_dim // 2
    
    # Extract slices
    lf_slice = lf_volume[:, :, middle_idx]
    hf_slice = hf_volume[:, :, middle_idx]
    
    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(lf_slice, cmap='gray')
    axes[0].set_title(f"Low Field (64mT) - Vol {volume_idx} - Slice {middle_idx}")
    axes[0].axis('off')
    
    axes[1].imshow(hf_slice, cmap='gray')
    axes[1].set_title(f"High Field (3T) - Vol {volume_idx} - Slice {middle_idx}")
    axes[1].axis('off')
    
    # plt.tight_layout()
    plt.show()

def main():
    # Define data directories
    lf_dir = "mri_resolution/train/low_field"
    hf_dir = "mri_resolution/train/high_field"
    
    # Generate file pairs
    pairs = make_pairs(lf_dir, hf_dir)
    
    if not pairs:
        print("No pairs found. Please check your data directories.")
        return

    print(f"Found {len(pairs)} MRI pairs. Loading data...")

    # Instantiate the dataset to handle loading and preprocessing.
    # We set patches_per_volume=1 because we aren't using the patch iterator,
    # but rather accessing the full cached volumes directly.
    dataset = MRIPatchDataset(pairs, patches_per_volume=1, cache_volumes=True)
    
    # Iterate through each volume in the dataset
    for i in range(len(pairs)):
        # MRIPatchDataset stores cached volumes. We can access the internal 
        # _get_volume_pair method to retrieve the full preprocessed (LF, HF, Mask) tuple.
        lf, hf, _ = dataset._get_volume_pair(i)
        
        print(f"Visualizing volume {i}...")
        visualize_middle_slice(lf, hf, i)

if __name__ == "__main__":
    main()