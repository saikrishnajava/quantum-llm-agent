#!/bin/bash
# Setup script for Linux + NVIDIA GPU (P2200 / CUDA)
# Run: bash setup_gpu.sh

set -e

echo "=== Quantum LLM Agent — GPU Setup ==="

# 1. Check NVIDIA driver
echo ""
echo "--- Checking NVIDIA GPU ---"
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers first."
    echo "  Ubuntu: sudo apt install nvidia-driver-535"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# 2. Check CUDA
echo ""
echo "--- Checking CUDA ---"
if ! command -v nvcc &> /dev/null; then
    echo "WARNING: nvcc not found. Install CUDA toolkit:"
    echo "  Ubuntu: sudo apt install nvidia-cuda-toolkit"
    echo "  Or: https://developer.nvidia.com/cuda-downloads"
fi

# 3. Create venv
echo ""
echo "--- Setting up Python environment ---"
python3 -m venv .venv
source .venv/bin/activate

# 4. Install dependencies
echo ""
echo "--- Installing dependencies ---"
pip install --upgrade pip
pip install pennylane pennylane-lightning numpy scipy pyyaml pytest

# 5. Install GPU acceleration
echo ""
echo "--- Installing GPU acceleration (pennylane-lightning-gpu) ---"
pip install pennylane-lightning[gpu] custatevec-cu12 || {
    echo ""
    echo "WARNING: GPU packages failed to install."
    echo "Falling back to CPU lightning.qubit (still fast, uses adjoint)."
    echo "For GPU: ensure CUDA 11.x or 12.x is properly installed."
}

# 6. Verify
echo ""
echo "--- Verification ---"
python3 -c "
import pennylane as qml
print(f'PennyLane: {qml.__version__}')

# Check available devices
devs = qml.plugin_devices
print(f'Available devices: {list(devs.keys())}')

# Test lightning.qubit
try:
    dev = qml.device('lightning.qubit', wires=4)
    print('lightning.qubit: OK')
except Exception as e:
    print(f'lightning.qubit: FAILED ({e})')

# Test lightning.gpu
try:
    dev = qml.device('lightning.gpu', wires=4)
    print('lightning.gpu: OK (GPU acceleration available!)')
except Exception as e:
    print(f'lightning.gpu: not available ({e})')

# Test CUDA via PyTorch (optional)
try:
    import torch
    print(f'PyTorch CUDA: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
except ImportError:
    print('PyTorch: not installed (not required)')
"

echo ""
echo "--- Setup complete ---"
echo "To run training: python examples/demos/train_and_benchmark.py"
echo "To run tests:    python -m pytest tests/ -v"
