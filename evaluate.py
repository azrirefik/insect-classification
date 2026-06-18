"""Evaluation: classification report, confusion matrix, per-class metrics."""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from config import C
from dataset import test_loader, class_names
from model import ARCHITECTURES
from utils import get_device


@torch.no_grad()
def evaluate(model, output_dir: Path):
    device = get_device()
    model = model.to(device).eval()

    all_preds, all_labels = [], []
    loader = test_loader()
    for x, y in tqdm(loader, desc="Evaluating"):
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())

    classes = class_names()
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    np.save(output_dir / "confusion_matrix.npy", cm)

    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"\nTest Accuracy: {acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=classes, zero_division=0))

    return report, cm, acc


def load_model(model_path: str | Path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    state = torch.load(str(model_path), map_location=device, weights_only=True)
    for arch_name in ["deep_cnn", "simple_cnn", "transfer_resnet18"]:
        model = ARCHITECTURES[arch_name]()
        try:
            model.load_state_dict(state)
            return model, arch_name
        except RuntimeError:
            continue
    raise RuntimeError(f"Could not match state dict to any architecture: {model_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Path to best_model.pt")
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    model, arch_name = load_model(args.model)
    print(f"Loaded {arch_name} from {args.model}")
    out = Path(args.output) if args.output else Path(args.model).parent / "eval"
    evaluate(model, out)
