import torch.nn as nn
from torchvision import models
from config import C


def _conv_block(in_ch: int, out_ch: int, bn: bool, dropout: float) -> nn.Sequential:
    layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True)]
    if bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.MaxPool2d(2))
    if dropout:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


class SimpleCNN(nn.Module):
    """Lightweight baseline — 3 conv blocks."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32, C.use_batchnorm, 0.0),
            _conv_block(32, 64, C.use_batchnorm, 0.0),
            _conv_block(64, 128, C.use_batchnorm, 0.0),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, C.num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class DeepCNN(nn.Module):
    """Deeper variant — 4 conv blocks, dropout."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32, C.use_batchnorm, 0.0),
            _conv_block(32, 64, C.use_batchnorm, 0.0),
            _conv_block(64, 128, C.use_batchnorm, 0.0),
            _conv_block(128, 256, C.use_batchnorm, 0.25 if C.use_dropout else 0.0),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5) if C.use_dropout else nn.Identity()
        self.classifier = nn.Linear(256, C.num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


class TransferResNet18(nn.Module):
    """ResNet18 backbone, frozen → fine-tuned classifier."""
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in self.backbone.parameters():
            param.requires_grad = False
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5) if C.use_dropout else nn.Identity(),
            nn.Linear(in_features, C.num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


ARCHITECTURES = {
    "simple_cnn": SimpleCNN,
    "deep_cnn": DeepCNN,
    "transfer_resnet18": TransferResNet18,
}
