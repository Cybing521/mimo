#!/bin/bash
# DRL Dependencies Installation Script
# Run this after activating your virtual environment

echo "================================================"
echo "Installing DRL dependencies for MA-MIMO..."
echo "================================================"

# Check if venv is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: Virtual environment not detected!"
    echo "Please activate venv first:"
    echo "  source venv/bin/activate"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install PyTorch (CPU version)
echo ""
echo "📦 Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Gym
echo ""
echo "📦 Installing Gym..."
pip install gym

# Install other dependencies
echo ""
echo "📦 Installing other dependencies..."
pip install tensorboard

# Verify installation
echo ""
echo "✅ Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import gym; print(f'Gym version: {gym.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

echo ""
echo "================================================"
echo "✅ Installation completed!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Test environment:    python drl/env.py"
echo "  2. Test networks:       python drl/networks.py"
echo "  3. Test agent:          python drl/agent.py"
echo "  4. Start training:      python experiments/train_drl.py --num_episodes 100"
echo ""

