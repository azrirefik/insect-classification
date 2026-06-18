"""Grad-CAM visualization — shows where the model looks."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from config import C
from dataset import class_names, test_loader

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor * IMAGENET_STD[:, None, None] + IMAGENET_MEAN[:, None, None]


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(x)
        score = logits[0, class_idx]
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = F.relu(cam)
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam.detach().cpu().numpy()


def generate_gradcam(model, output_dir: Path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device).eval()

    loader = test_loader()
    x_batch, y_batch = next(iter(loader))
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

    target_layer = None
    for layer in [model.features[-2] if hasattr(model, "features") else None,
                  model.backbone.layer4[-1].conv2 if hasattr(model, "backbone") else None]:
        if layer is not None:
            target_layer = layer
            break
    if target_layer is None:
        print("Skipping Grad-CAM: no conv layer found")
        return

    gradcam = GradCAM(model, target_layer)
    classes = class_names()
    output_dir.mkdir(parents=True, exist_ok=True)

    n_samples = min(5, len(x_batch))
    fig, axes = plt.subplots(n_samples, 2, figsize=(8, 3 * n_samples))
    if n_samples == 1:
        axes = [axes]

    for i in range(n_samples):
        x = x_batch[i:i + 1]
        true_label = y_batch[i].item()
        logit = model(x)
        pred_label = logit.argmax(dim=1).item()

        cam = gradcam(x, pred_label)
        img = denormalize(x_batch[i].cpu()).permute(1, 2, 0).clamp(0, 1).numpy()
        cam_resized = transforms.Resize((C.img_size, C.img_size))(
            torch.tensor(cam).unsqueeze(0).unsqueeze(0)
        ).squeeze().numpy()

        axes[i][0].imshow(img)
        axes[i][0].set_title(f"True: {classes[true_label]}")
        axes[i][0].axis("off")
        axes[i][1].imshow(img)
        axes[i][1].imshow(cam_resized, cmap="jet", alpha=0.5)
        axes[i][1].set_title(f"Grad-CAM → {classes[pred_label]}")
        axes[i][1].axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "gradcam_samples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grad-CAM saved to {output_dir / 'gradcam_samples.png'}")


if __name__ == "__main__":
    import argparse
    from evaluate import load_model
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()
    model, arch_name = load_model(args.model)
    print(f"Loaded {arch_name} from {args.model}")
    out = Path(args.output) if args.output else Path(args.model).parent / "gradcam"
    generate_gradcam(model, out)
