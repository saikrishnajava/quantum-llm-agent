#!/usr/bin/env bash
# ============================================================
# Quantum LLM Agent — Environment Setup Script
# ============================================================
# Usage:
#   chmod +x setup.sh && ./setup.sh
# ============================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"

echo "========================================"
echo " Quantum LLM Agent — Setup"
echo "========================================"

# 1. Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "[1/4] Virtual environment exists."
fi

# 2. Upgrade pip
echo "[2/4] Upgrading pip..."
"$PIP" install --upgrade pip --quiet

# 3. Install dependencies
echo "[3/4] Installing dependencies..."
"$PIP" install -e "${PROJECT_DIR}" --quiet
"$PIP" install pytest --quiet

# 4. Verify installation
echo "[4/4] Verifying installation..."
"$PYTHON" -c "
import pennylane as qml
import numpy as np
import yaml

print(f'  PennyLane: {qml.version()}')
print(f'  NumPy:     {np.__version__}')

# Quick quantum sanity check
dev = qml.device('default.qubit', wires=4)
@qml.qnode(dev)
def test_circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))
result = test_circuit()
print(f'  Quantum test: {result:.4f} (expected ~0.0)')
print('  All checks passed!')
"

echo ""
echo "Setup complete! Activate with:"
echo "  source .venv/bin/activate"
echo ""
echo "Run tests:"
echo "  python -m pytest tests/ -v"
echo ""
echo "Run 12-qubit PoC demo:"
echo "  python examples/demos/poc_12qubit.py"
