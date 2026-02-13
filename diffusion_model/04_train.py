"""
STEP 4: Training Loop for Diffusion Model

This brings together:
1. Dataset (from 02_dataset.py)
2. Model (from 03_model.py)
3. Diffusion schedule (from 01_schedule.py)
4. Training objective (noise prediction)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm


def train_diffusion_epoch(
    model,
    schedule,
    loader,
    optimizer,
    device,
    scaler=None,
):
    """
    Train diffusion model for one epoch.
    
    Args:
        model: ConditionalUNet3D
        schedule: DiffusionSchedule
        loader: DataLoader with {'condition': ..., 'target': ...}
        optimizer: torch optimizer
        device: 'cuda' or 'cpu'
        scaler: GradScaler for mixed precision (optional)
    
    Returns:
        average loss for the epoch
    """
    model.train()
    schedule = schedule.to(device)
    
    total_loss = 0.0
    num_batches = 0
    
    # Progress bar
    pbar = tqdm.tqdm(loader, desc="Training")
    
    for batch in pbar:
        condition = batch['condition'].to(device)  # (B, 2, D, H, W)
        target = batch['target'].to(device)        # (B, 1, D, H, W)
        
        batch_size = target.shape[0]
        
        # 1. Sample random timesteps
        t = torch.randint(
            0, schedule.timesteps, 
            (batch_size,), 
            device=device
        )
        
        # 2. Sample random noise
        noise = torch.randn_like(target)
        
        # 3. Add noise to target (forward process)
        x_t = schedule.add_noise(target, t, noise)
        
        # 4. Zero gradients
        optimizer.zero_grad(set_to_none=True)
        
        # 5. Forward pass: predict noise
        with torch.autocast(device_type=device, enabled=(scaler is not None)):
            noise_pred = model(x_t, condition, t)
            
            # 6. Compute loss (noise matching)
            loss = F.mse_loss(noise_pred, noise)
        
        # 7. Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item():.4f})
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def validate_diffusion(
    model,
    schedule,
    loader,
    device,
):
    """
    Validate diffusion model (no sampling, just noise prediction loss).
    
    Args:
        model: ConditionalUNet3D
        schedule: DiffusionSchedule
        loader: DataLoader
        device: 'cuda' or 'cpu'
    
    Returns:
        average validation loss
    """
    model.eval()
    schedule = schedule.to(device)
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm.tqdm(loader, desc="Validation")
    
    for batch in pbar:
        condition = batch['condition'].to(device)
        target = batch['target'].to(device)
        
        batch_size = target.shape[0]
        t = torch.randint(0, schedule.timesteps, (batch_size,), device=device)
        noise = torch.randn_like(target)
        x_t = schedule.add_noise(target, t, noise)
        
        noise_pred = model(x_t, condition, t)
        loss = F.mse_loss(noise_pred, noise)
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': loss.item():.4f})
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


# ============================================================================
# EXAMPLE: FULL TRAINING SCRIPT
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("STEP 4: TRAINING LOOP TUTORIAL")
    print("=" * 70)
    
    print("\n1. Training Objective (Noise Prediction):")
    print("-" * 70)
    print("""
    The model learns through a simple 3-step process per batch:
    
    1. FORWARD PROCESS: Add noise to target
       noise = random_noise
       x_t = sqrt(α̅ₜ) * hf + sqrt(1-α̅ₜ) * noise
    
    2. MODEL: Predict the noise that was added
       noise_pred = model(x_t, condition, t)
    
    3. LOSS: Compare prediction to actual noise
       loss = MSE(noise_pred, noise)
       
    By predicting noise at each timestep t ∈ [0, 1000], the model
    learns a powerful image transformation function.
    """)
    
    print("\n2. Why This Works:")
    print("-" * 70)
    print("""
    - At t=0: x_t ≈ hf, so model learns fine details
    - At t=999: x_t ≈ noise, so model learns coarse structure
    - Covering all timesteps = covering all scales
    
    During inference (sampling):
    - Start from pure noise
    - Iteratively predict + remove noise
    - Result: refined MRI
    """)
    
    print("\n3. Full Training Loop Structure:")
    print("-" * 70)
    print("""
    from diffusers.optimization import get_cosine_schedule_with_warmup
    
    model = SimpleConditionalUNet3D(...)
    schedule = LinearSchedule(timesteps=1000)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    num_epochs = 50
    num_warmup_steps = len(train_loader) * 2  # first 2 epochs
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps,
        num_training_steps=len(train_loader) * num_epochs
    )
    
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    for epoch in range(num_epochs):
        train_loss = train_diffusion_epoch(
            model, schedule, train_loader, optimizer, device, scaler
        )
        
        val_loss = validate_diffusion(
            model, schedule, val_loader, device
        )
        
        lr_scheduler.step()
        
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Save checkpoint
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'schedule': schedule,
            }, f'diffusion_epoch_{epoch:03d}.pt')
    """)
    
    print("\n4. Key Training Tips:")
    print("-" * 70)
    print("""
    ✓ Mixed precision training (AMP) helps speed up training
    ✓ Gradient clipping prevents exploding gradients
    ✓ Learning rate warmup stabilizes early training
    ✓ Cosine annealing schedule is good for diffusion models
    ✓ EMA (Exponential Moving Average) checkpoint works well
    ✓ Validate frequently (use fast loss, not full sampling)
    ✓ Sample during training to visualize progress
    """)
    
    print("\n5. Hyperparameters to Tune:")
    print("-" * 70)
    print("""
    Learning rate:     1e-4 to 1e-3 (try 2e-4 first)
    Batch size:        4-16 (depends on GPU VRAM)
    Num epochs:        20-100 (start with 20)
    Warmup steps:      5-10% of total training steps
    Gradient clip:     1.0 (prevent explosions)
    Weight decay:      0.01 (L2 regularization)
    EMA decay:         0.9999 (for exponential moving average)
    """)
    
    print("\n" + "=" * 70)
    print("Next: Open 05_inference.py to sample from trained model")
    print("=" * 70)
