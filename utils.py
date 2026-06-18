import random
import time
from contextlib import contextmanager

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@contextmanager
def timer(label: str = ""):
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    mins, secs = divmod(elapsed, 60)
    print(f"[timer] {label} {int(mins)}m {secs:.1f}s")
