# Insect Classification

My final year project at University of Nottingham. I wanted to understand why transfer learning crushes custom architectures on small datasets, so I built three CNNs and made them fight.

| Architecture | Val Accuracy | Trainable Params |
|---|---|---|
| SimpleCNN | 62.4% | 94,470 |
| DeepCNN | 58.2% | 391,430 |
| **Transfer ResNet-18** | **81.1%** | 3,078 |

The interesting finding: DeepCNN does *worse* than SimpleCNN despite having 4x the parameters. More capacity doesn't help when your dataset is 2,500 32x32 images. ResNet-18 wins with only 3K trainable params because ImageNet pre-training already learned useful edge and texture features.

## What's in here

Everything from data loading to a compiled PDF:

- `model.py` - three architectures: SimpleCNN (3 conv blocks), DeepCNN (4 blocks + dropout), TransferResNet18 (frozen backbone, trainable classifier)
- `ablation.py` - toggles augmentation/dropout/batchnorm to isolate what actually matters. Augmentation contributes +13%, batchnorm +9%.
- `grad_cam.py` - Grad-CAM heatmaps showing where the model looks. Useful for catching background bias.
- `hyperparameter_search.py` - grid search over learning rates
- `paper/paper.tex` - 30-page LaTeX research paper with 20 citations. Auto-filled from experiment outputs via `build_paper.py`.
- `run_experiments.sh` - runs the entire pipeline overnight. Trains all 3 architectures, runs hyperparameter search, ablation, evaluation, and Grad-CAM.

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 download_dataset.py    # extracts 5 insect classes from CIFAR-100
python3 train.py --arch simple_cnn
```

Or just run everything:

```bash
bash run_experiments.sh    # ~4 hours on M1 Pro MPS
```

## What I learned

- CIFAR-100's 32x32 resolution is brutal for fine-grained classification. These insects are basically colored blobs at that resolution.
- Early stopping with patience=10 saved me from overfitting DeepCNN, which peaked at epoch 27 then declined.
- Grad-CAM revealed that SimpleCNN sometimes looks at background color instead of the insect. Transfer learning mostly fixes this.
- Writing the paper took longer than writing the code.

## What I'd do differently

- Use a higher-resolution dataset. CIFAR-100 was convenient but the resolution ceiling limits all architectures.
- Try a lighter transfer model (MobileNetV2) to see if the win is about pre-training or architecture.
- Actually run the hyperparameter search on all three architectures instead of just DeepCNN.

## License

MIT
