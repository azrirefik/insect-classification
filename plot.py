"""Plotting utilities — saves PNGs to output directory."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_history(history: dict, save_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch"), ax1.set_ylabel("Loss"), ax1.legend(), ax1.set_title("Loss")
    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Val")
    ax2.set_xlabel("Epoch"), ax2.set_ylabel("Accuracy"), ax2.legend(), ax2.set_title("Accuracy")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], save_path: Path):
    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.6), max(6, len(class_names) * 0.5)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))), ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(class_names))), ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted"), ax.set_ylabel("True"), ax.set_title("Confusion Matrix")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm[i,j]:.0f}" if cm[i, j] > 0 else "", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_comparison(results: dict, save_path: Path):
    """results = {label: {'train_acc': [...], 'val_acc': [...]}}"""
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, hist in results.items():
        ax.plot(hist["val_acc"], label=label)
    ax.set_xlabel("Epoch"), ax.set_ylabel("Validation Accuracy"), ax.set_title("Ablation Study — Val Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_lr_comparison(results: dict, save_path: Path):
    """results = {lr: {'val_acc': [...]}}"""
    fig, ax = plt.subplots(figsize=(10, 6))
    for lr, hist in results.items():
        ax.plot(hist["val_acc"], label=f"lr={lr}")
    ax.set_xlabel("Epoch"), ax.set_ylabel("Validation Accuracy"), ax.set_title("Learning Rate Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
