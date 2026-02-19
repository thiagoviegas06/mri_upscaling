# MRI Upscaling

## Overview
Low-field to high-field MRI super-resolution using a 2.5D UNet (stack of 7 slices, base channels 128) trained to maximize MS-SSIM.

## Model Workflow

### Architecture
- **2.5D UNet**: Takes 7 adjacent LF slices as input channels (center ±3) and predicts the center HF slice
- **Input**: `(B, 7, H, W)` — stack of 7 axial slices from low-field MRI
- **Output**: `(B, 1, H, W)` — single enhanced high-field slice
- **Base channels**: 128 with residual blocks and attention gates

### Training Strategy
1. **Dataset**: 2.5D patch-based sampling
   - Extracts 96×96 XY patches from random Z positions
   - Stacks 7 adjacent LF slices as channels
   - Target: center HF slice
   - Augmentation: in-plane flips and 90° rotations
   - Foreground sampling: biases patches toward brain tissue

2. **Loss Function**: MS-SSIM + L1 (optionally MSE)
   - Multi-Scale SSIM (aligned with evaluation metric)
   - Warmup schedule: 0.2 → 0.6 MS-SSIM weight over 5 epochs
   - Final: `0.4 * L1 + 0.6 * (1 - MS-SSIM)`
   - Optionally, MSE can be added as an additional term for experimentation

3. **Training Details**
   - Optimizer: AdamW (lr=2e-4, weight_decay=1e-4)
   - EMA: exponential moving average (decay=0.999)
   - Scheduler: ReduceLROnPlateau on validation score
   - Early stopping: patience=12 epochs
   - Mixed precision (AMP) on CUDA

4. **Validation**
   - Full-volume inference with sliding window (patch_size=96, stride=48)
   - 2D Gaussian-weighted patch blending per slice
   - Metrics: SSIM, MS-SSIM, PSNR per slice
   - Volume cache: preprocessed validation volumes cached across epochs

### Inference
- Slice-by-slice prediction with 2.5D stacks
- For each Z position:
  - Extract 5-slice stack centered at Z
  - Sliding 96×96 patches with stride=48
  - Gaussian blending in XY plane
- Handles edge cases with clamping at volume boundaries

### File Structure
```
preprocessing.py         # 2.5D dataset with slice stacking and normalization
model.py                 # UNet2.5D with attention gates (base=128, stack_size=7)
train.py                 # Training loop, MS-SSIM + L1 loss, validation
main.py                  # Training script with warmup and config
mri_resolution/metric.py # MS-SSIM and evaluation metrics
test.py                  # Inference and testing script
```

### Evaluation Metric
The competition uses **MS-SSIM** (Multi-Scale SSIM) computed per 2D slice:
- 5 scales with Gaussian windows (11×11, σ=1.5)
- Standard weights: [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
- See `mri_resolution/metric.py` for reference implementation

### Running Training
```bash
sbatch slurm_job_metric.sh
```
Trains on HPC with:
- H200 GPU (48hrs max)
- 32GB RAM, 4 CPUs
- Checkpoints saved to `$MODEL_DESTINATION`
