from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    # paths
    data_root: Path = ROOT / "data"
    output_root: Path = ROOT / "outputs"
    # training
    batch_size: int = 32
    epochs: int = 50
    learning_rates: list[float] = field(default_factory=lambda: [1e-2, 1e-3, 1e-4])
    weight_decay: float = 1e-4
    # image
    img_size: int = 224
    # ablation flags
    use_augmentation: bool = True
    use_dropout: bool = True
    use_batchnorm: bool = True
    # early stopping
    patience: int = 10
    # reproducibility
    seed: int = 42

    num_classes: int = 6  # default, updated when dataset loads


C = Config()
