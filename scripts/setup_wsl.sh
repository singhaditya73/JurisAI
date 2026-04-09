#!/bin/bash
# =============================================================================
# JurisAI - WSL Environment Setup & Training
# Run from Windows: wsl -d Ubuntu -- bash /mnt/c/Users/ADITYA\ SINGH/Desktop/JurisAI/scripts/setup_wsl.sh
# =============================================================================

set -e

PROJ_DIR="/mnt/c/Users/ADITYA SINGH/Desktop/JurisAI"
VENV_DIR="$PROJ_DIR/venv_wsl"

echo "================================================="
echo "  JurisAI - WSL Setup"
echo "================================================="

# Step 1: Create venv
echo ""
echo "[1/4] Setting up Python virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "  Created: $VENV_DIR"
else
    echo "  Already exists."
fi
source "$VENV_DIR/bin/activate"

# Step 2: Install pip and core deps
echo ""
echo "[2/4] Installing PyTorch + CUDA..."
pip install --upgrade pip -q
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q

# Verify CUDA
python3 -c "import torch; print(f'  PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Step 3: Install Unsloth + dependencies
echo ""
echo "[3/4] Installing Unsloth + dependencies..."
pip install unsloth -q
pip install datasets transformers accelerate peft bitsandbytes trl -q
pip install pandas numpy PyYAML rich tqdm rouge-score nltk scikit-learn tensorboard -q

# Step 4: Verify
echo ""
echo "[4/4] Verifying installation..."
python3 -c "
from unsloth import FastLanguageModel
import torch
print(f'  Unsloth: OK')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
print(f'  GPU: {torch.cuda.get_device_name(0)}')
print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

echo ""
echo "================================================="
echo "  Setup complete!"
echo "================================================="
echo ""
echo "To train, run:"
echo "  wsl -d Ubuntu -- bash -c 'cd \"$PROJ_DIR\" && source venv_wsl/bin/activate && python -m src.training.pretrain'"
echo ""
