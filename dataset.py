import json

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import C
from utils import get_device

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(augment: bool = True):
    t = [transforms.Resize((C.img_size, C.img_size))]
    if augment:
        t += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    t += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(t)


def _loader(split: str, augment: bool, shuffle: bool) -> DataLoader:
    path = C.data_root / split
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run download_dataset.py first.")
    dataset = datasets.ImageFolder(path, transform=build_transforms(augment))
    return DataLoader(dataset, batch_size=C.batch_size, shuffle=shuffle, num_workers=2)


def train_loader() -> DataLoader:
    return _loader("train", augment=C.use_augmentation, shuffle=True)


def val_loader() -> DataLoader:
    return _loader("val", augment=False, shuffle=False)


def test_loader() -> DataLoader:
    return _loader("test", augment=False, shuffle=False)


def class_names() -> list[str]:
    path = C.data_root / "train"
    if not path.exists():
        return []
    return sorted(d.name for d in path.iterdir() if d.is_dir())


def class_distribution() -> dict:
    """Report number of images per class. Flags data imbalance."""
    dist = {}
    for split in ("train", "val", "test"):
        path = C.data_root / split
        if not path.exists():
            continue
        ds = datasets.ImageFolder(path)
        for cls_name, count in zip(ds.classes, torch.bincount(torch.tensor(ds.targets)).tolist()):
            dist.setdefault(cls_name, {})[split] = count
    return dist


if __name__ == "__main__":
    device = get_device()
    print(f"Device: {device}")
    try:
        loader = train_loader()
        x, y = next(iter(loader))
        print(f"Batch shape: {x.shape}, classes: {len(class_names())}")
        print("Class distribution:")
        print(json.dumps(class_distribution(), indent=2))
    except FileNotFoundError as e:
        print(e)
