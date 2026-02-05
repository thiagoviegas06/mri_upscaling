import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

def train_one_epoch(model, loader, optim, device, scaler):
    model.train()
    running = 0.0

    for lf, hf in loader:
        lf = lf.to(device, non_blocking=True)
        hf = hf.to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)

        with autocast():
            pred = model(lf)
            loss = F.l1_loss(pred, hf)

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        running += loss.item()

    return running / max(1, len(loader))

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    running = 0.0

    for lf, hf in loader:
        lf = lf.to(device, non_blocking=True)
        hf = hf.to(device, non_blocking=True)

        with autocast():
            pred = model(lf)
            loss = F.l1_loss(pred, hf)

        running += loss.item()

    return running / max(1, len(loader))