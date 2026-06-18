"""Hyperparameter search — grid search over learning rates."""
import json
from datetime import datetime
from pathlib import Path

from config import C
from model import ARCHITECTURES
from train import train


def run_search(arch_name: str, lrs: list[float] | None = None) -> dict:
    lrs = lrs or C.learning_rates
    base_dir = C.output_root / f"lr_search_{arch_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for lr in lrs:
        run_dir = base_dir / f"lr_{lr}"
        run_dir.mkdir(exist_ok=True)
        model = ARCHITECTURES[arch_name]()
        history = train(model, lr, run_dir, label=f"{arch_name}_lr{lr}")
        results[str(lr)] = {
            "best_val_acc": history["best_val_acc"],
            "best_epoch": history["best_epoch"],
            "n_params": history["n_params"],
        }

    best_lr = max(results, key=lambda k: results[k]["best_val_acc"])
    print(f"\nBest LR: {best_lr} (val_acc={results[best_lr]['best_val_acc']:.4f})")

    with open(base_dir / "lr_results.json", "w") as f:
        json.dump({"best_lr": best_lr, "results": results}, f, indent=2)

    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="simple_cnn", choices=list(ARCHITECTURES))
    ap.add_argument("--lrs", type=float, nargs="+", default=None)
    args = ap.parse_args()
    run_search(args.arch, args.lrs)
