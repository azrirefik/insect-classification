"""Ablation studies — toggle augmentation, dropout, batchnorm."""
import json
from datetime import datetime
from pathlib import Path

from config import C
from model import ARCHITECTURES
from train import train


ABLATION_CONFIGS = {
    "baseline":        {"aug": True,  "dropout": True,  "bn": True},
    "no_augmentation": {"aug": False, "dropout": True,  "bn": True},
    "no_dropout":      {"aug": True,  "dropout": False, "bn": True},
    "no_batchnorm":    {"aug": True,  "dropout": True,  "bn": False},
}


def run_ablation(arch_name: str, lr: float = 1e-3):
    base_dir = C.output_root / f"ablation_{arch_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for label, flags in ABLATION_CONFIGS.items():
        C.use_augmentation = flags["aug"]
        C.use_dropout = flags["dropout"]
        C.use_batchnorm = flags["bn"]

        run_dir = base_dir / label
        run_dir.mkdir(exist_ok=True)
        model = ARCHITECTURES[arch_name]()
        history = train(model, lr, run_dir, label=f"{arch_name}_{label}")
        results[label] = {
            "best_val_acc": history["best_val_acc"],
            "best_epoch": history["best_epoch"],
            "val_acc_curve": history["val_acc"],
        }

    # Restore defaults
    C.use_augmentation = True
    C.use_dropout = True
    C.use_batchnorm = True

    best = max(results, key=lambda k: results[k]["best_val_acc"])
    print(f"\nBest ablation config: {best} (val_acc={results[best]['best_val_acc']:.4f})")

    with open(base_dir / "ablation_results.json", "w") as f:
        json.dump({"best_config": best, "results": results}, f, indent=2)

    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="deep_cnn", choices=list(ARCHITECTURES))
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    run_ablation(args.arch, args.lr)
