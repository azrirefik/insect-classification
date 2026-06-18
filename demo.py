"""Self-check: builds all architectures, forward-passes random input."""
import torch

from config import C
from model import ARCHITECTURES
from utils import get_device, count_params

if __name__ == "__main__":
    device = get_device()
    print(f"Device: {device}")
    C.num_classes = 6

    dummy = torch.randn(2, 3, C.img_size, C.img_size).to(device)

    for name, builder in ARCHITECTURES.items():
        model = builder().to(device)
        with torch.no_grad():
            out = model(dummy)
        n = count_params(model)
        print(f"{name:<20s} params: {n:>10,}  output: {tuple(out.shape)}  device: {device}")

    print("\nAll architectures OK.")
