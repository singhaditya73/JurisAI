#!/bin/bash
# =============================================================================
# JurisAI - Full Training Pipeline (Stage 1 + Stage 2)
# Runs pretraining then finetuning automatically.
# =============================================================================

set -e

PROJ_DIR="/mnt/c/Users/ADITYA SINGH/Desktop/JurisAI"
cd "$PROJ_DIR"
source venv_wsl/bin/activate

echo "================================================="
echo "  JurisAI - Full Training Pipeline"
echo "  Started: $(date)"
echo "================================================="

# Stage 1: Continual Pretraining
echo ""
echo "[STAGE 1] Continual Pretraining..."
echo "  Start: $(date)"
python3 -m src.training.pretrain
echo "  Done:  $(date)"

# Stage 2: Instruction Fine-Tuning
echo ""
echo "[STAGE 2] Instruction Fine-Tuning..."
echo "  Start: $(date)"
python3 -m src.training.finetune --from-pretrained ./models/adapters/pretrain_v1/final
echo "  Done:  $(date)"

echo ""
echo "================================================="
echo "  ALL TRAINING COMPLETE!"
echo "  Finished: $(date)"
echo "================================================="
echo ""
echo "Next: python3 -m src.inference.generate"
