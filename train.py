"""Training loop. Call train() with a model + config, returns history dict."""
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from config import C
from dataset import train_loader, val_loader, class_names
from utils import get_device, count_params, timer


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        _, preds = torch.max(model(x) if total == 0 else model(x), 1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total


def train(model: nn.Module, lr: float, output_dir: Path, label: str = "") -> dict:
    device = get_device()
    model = model.to(device)
    n_params = count_params(model)
    print(f"\n{'='*60}")
    print(f"Training: {label} | params: {n_params:,} | lr: {lr} | device: {device}")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=C.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    train_ldr = train_loader()
    val_ldr = val_loader()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": lr}
    best_val_acc, best_epoch, patience_counter = 0.0, 0, 0
    best_state = None

    with timer(f"{label} training"):
        for epoch in range(C.epochs):
            train_loss, train_acc = train_one_epoch(model, train_ldr, optimizer, criterion, device)
            val_loss, val_acc = validate(model, val_ldr, criterion, device)
            scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(f"Epoch {epoch+1:3d}/{C.epochs} | "
                  f"tr_loss: {train_loss:.4f} tr_acc: {train_acc:.4f} | "
                  f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc, best_epoch, patience_counter = val_acc, epoch, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, output_dir / "best_model.pt")
            else:
                patience_counter += 1

            if patience_counter >= C.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    history["best_epoch"] = best_epoch
    history["best_val_acc"] = best_val_acc
    history["n_params"] = n_params
    history["n_classes"] = len(class_names())
    history["classes"] = class_names()

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=float)

    print(f"Best val acc: {best_val_acc:.4f} at epoch {best_epoch+1}")
    return history


if __name__ == "__main__":
    import argparse
    from model import ARCHITECTURES

    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="simple_cnn", choices=list(ARCHITECTURES))
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    run_dir = Path(args.output) if args.output else C.output_root / f"train_{args.arch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = ARCHITECTURES[args.arch]()
    train(model, args.lr, run_dir, label=args.arch)
