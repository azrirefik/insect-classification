#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="outputs/experiment_${TIMESTAMP}.log"
mkdir -p outputs
exec > >(tee -a "$LOG") 2>&1

echo "=== Insect Classification FYP — Autonomous Pipeline ==="
echo "Started: $(date)"
echo "Machine: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, MPS: {torch.backends.mps.is_available()}')"

echo ""
echo "[1/5] Downloading dataset..."
python3 download_dataset.py

echo ""
echo "[2/5] Architecture comparison..."
python3 train.py --arch simple_cnn --output outputs/arch_simple_cnn
python3 train.py --arch deep_cnn --output outputs/arch_deep_cnn
python3 train.py --arch transfer_resnet18 --output outputs/arch_resnet18

echo ""
echo "[3/5] Hyperparameter search (on best architecture)..."
python3 hyperparameter_search.py --arch deep_cnn

echo ""
echo "[4/5] Ablation studies (on best architecture)..."
python3 ablation.py --arch deep_cnn

echo ""
echo "[5/5] Final evaluation + Grad-CAM..."
for arch in simple_cnn deep_cnn transfer_resnet18; do
    model_path="outputs/arch_${arch}/best_model.pt"
    if [ -f "$model_path" ]; then
        python3 evaluate.py --model "$model_path" --output "outputs/arch_${arch}/eval"
        python3 grad_cam.py --model "$model_path" --output "outputs/arch_${arch}/gradcam" || echo "Grad-CAM skipped for ${arch}"
    fi
done

echo ""
echo "=== Pipeline complete: $(date) ==="
echo "Log: $LOG"
echo ""
echo "Next: run 'python3 build_paper.py' to auto-fill the LaTeX paper,"
echo "then 'cd paper && tectonic fyp_paper.tex' to compile the PDF."
