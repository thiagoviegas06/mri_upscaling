# Step-by-Step Diffusion Model Fine-Tuning Tutorial

## Overview

We'll build a **Conditional Diffusion Model** for MRI refinement using HuggingFace `diffusers` library.

**What is Diffusion?**
- Start with noise → gradually add signal (forward)
- Start with noise → gradually remove noise (reverse) ← what we train

**Goal:** Given low-field MRI (lf) and stage1 prediction (y1), predict high-field MRI (hf)

---

## Step 1: Install & Setup

```bash
pip install diffusers transformers accelerate datasets
```

Key libraries:
- `diffusers`: provides UNet2D/3D, schedulers, pipelines
- `accelerate`: multi-GPU training support
- `datasets`: easy data loading

---

## Step 2: Understanding the Diffusion Schedule

The forward process (adding noise) follows a schedule:
- **Beta schedule**: how much noise to add at each step
- **Alpha schedule**: how much signal to keep

Common schedules:
- Linear: simple, works okay
- Cosine: smooth, often better
- Exponential: aggressive noise

```python
# Example: Linear schedule
timesteps = 1000
beta_start = 0.0001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, timesteps)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)  # cumulative product

# At timestep t:
# x_t = sqrt(alpha_cumprod[t]) * x_0 + sqrt(1 - alpha_cumprod[t]) * noise
```

---

## Step 3: The Noise Prediction Model

We need a neural network that:
- **Input:** noisy image + condition + timestep
- **Output:** predicted noise

Architecture options:
1. **UNet3D** (simple, good for 3D): skip connections, downsampling/upsampling
2. **DiT** (Diffusion Transformer): attention-based, might be overkill
3. **Conditional UNet** (recommended): UNet that accepts condition

For our task, we'll use HuggingFace's `UNet3DConditionModel`:
- Designed for conditional generation
- Cross-attention with text embeddings → we'll use image features
- Flexible, battle-tested

---

## Step 4: The Training Objective

**Noise Matching (Denoising Score Matching):**

```
1. Sample timestep t uniformly
2. Add noise to target: x_t = sqrt(α̅ₜ) * x₀ + sqrt(1-α̅ₜ) * ε
3. Model predicts noise: ε̂ = model(x_t, cond, t)
4. Loss = MSE(ε̂, ε)
```

This teaches the model to predict what noise was added.

---

## Step 5: Inference (Sampling)

**DDPM Sampling:** Reverse the noise-adding process

```
x₀ ← x₁ ← x₂ ← ... ← x_T (pure noise)
```

At each step t → t-1:
```
x_{t-1} = (x_t - sqrt(1-α̅ₜ)/sqrt(1-α̅_{t-1}) * ε̂) / sqrt(α_t) + noise
```

Fewer steps = faster, but lower quality.

---

## What We'll Build

| File | Purpose |
|------|---------|
| `01_schedule.py` | Understand diffusion schedule |
| `02_dataset.py` | Custom conditional dataset |
| `03_model.py` | Build conditional UNet |
| `04_train.py` | Training loop |
| `05_inference.py` | Sampling & refinement |
| `config.py` | Centralized config |

---

## Next: Start with Step 1!

Open `01_schedule.py` and run it to see how the diffusion schedule works.
