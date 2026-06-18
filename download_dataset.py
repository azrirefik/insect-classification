"""Extract insect classes from CIFAR-100 (assumes data already downloaded)."""
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm

from config import C

INSECT_CLASSES = ["bee", "beetle", "butterfly", "caterpillar", "cockroach"]
CIFAR100_LABELS = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark",
    "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel",
    "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone",
    "television", "tiger", "tractor", "train", "trout", "tulip", "turtle",
    "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
]


def _make_splits(data_dir: Path, train_ratio=0.7, val_ratio=0.15, seed=42):
    np.random.seed(seed)
    for cls_dir in sorted(data_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        images = sorted(list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg")))
        if not images:
            continue
        np.random.shuffle(images)
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        splits = {"train": images[:n_train], "val": images[n_train:n_train + n_val], "test": images[n_train + n_val:]}
        for split_name, split_images in splits.items():
            split_path = C.data_root / split_name / cls_dir.name
            split_path.mkdir(parents=True, exist_ok=True)
            for img_path in split_images:
                shutil.copy(img_path, split_path / img_path.name)


def download():
    C.data_root.mkdir(parents=True, exist_ok=True)
    train_dir = C.data_root / "train"
    if train_dir.exists() and any(train_dir.iterdir()):
        classes = sorted(d.name for d in train_dir.iterdir() if d.is_dir())
        print(f"Dataset already exists at {C.data_root}")
        print(f"Classes ({len(classes)}): {', '.join(classes)}")
        C.num_classes = len(classes)
        return

    insect_indices = [i for i, label in enumerate(CIFAR100_LABELS) if label in INSECT_CLASSES]
    print(f"Extracting insect classes from CIFAR-100: {INSECT_CLASSES}")

    raw_dir = C.data_root.parent / "_cifar100_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    full_train = datasets.CIFAR100(root=str(C.data_root.parent), train=True, download=False)
    full_test = datasets.CIFAR100(root=str(C.data_root.parent), train=False, download=False)

    for idx in tqdm(range(len(full_train)), desc="Extracting train"):
        img, label = full_train[idx]
        if label in insect_indices:
            cls_name = CIFAR100_LABELS[label]
            out_dir = raw_dir / cls_name
            out_dir.mkdir(parents=True, exist_ok=True)
            img.save(out_dir / f"train_{idx:05d}.png")

    for idx in tqdm(range(len(full_test)), desc="Extracting test"):
        img, label = full_test[idx]
        if label in insect_indices:
            cls_name = CIFAR100_LABELS[label]
            out_dir = raw_dir / cls_name
            out_dir.mkdir(parents=True, exist_ok=True)
            img.save(out_dir / f"test_{idx:05d}.png")

    print("Splitting into train/val/test...")
    _make_splits(raw_dir, train_ratio=0.7, val_ratio=0.15)
    shutil.rmtree(raw_dir, ignore_errors=True)

    classes = sorted(d.name for d in train_dir.iterdir() if d.is_dir())
    print(f"Classes ({len(classes)}): {', '.join(classes)}")
    C.num_classes = len(classes)
    print(f"num_classes set to {C.num_classes}")


if __name__ == "__main__":
    download()
