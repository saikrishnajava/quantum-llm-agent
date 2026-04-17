# Quantum LLM Agent - Cross-Machine Run Log

This document tracks execution environments, bugs encountered, and performance metrics across different machines (e.g., MacBook vs. Ubuntu Desktop) to maintain shared knowledge across development environments.

## Run 1: Ubuntu Linux (NVIDIA GPU Setup)
**Date:** April 17, 2026
**Environment:** Ubuntu Linux, Python 3.12, CUDA 12
**Goal:** Run `setup_gpu.sh` and execute `examples/demos/train_gpu.py`.

### 1. Setup & Environment Issues Encountered
During the execution of `setup_gpu.sh`, the following package versions were resolved:
*   `pennylane` v0.44.1
*   `pennylane-lightning-gpu` v0.44.0
*   `numpy` v2.x

**Bug 1: PennyLane Device Plugin Syntax**
*   **Issue:** `setup_gpu.sh` verification failed with `TypeError: 'dict' object is not callable` on `qml.plugin_devices()`.
*   **Fix:** In PennyLane 0.44.1, this was changed to a property. Updated `setup_gpu.sh` line 58 to `devs = qml.plugin_devices`.

**Bug 2: NVIDIA GPU Backend Initialization**
*   **Issue:** Calling `qml.device('lightning.gpu', wires=4)` throws a C++ binding error: `Unable to convert function return value to a Python type! The signature was get_gpu_arch(device_number: int = 0) -> std::pair<int, int>`.
*   **Status:** Unresolved. Likely a version mismatch between `pennylane-lightning-gpu` 0.44.0 and the local CUDA/NVIDIA driver configuration. 
*   **Workaround:** The training script automatically falls back to `lightning.qubit` (C++ CPU adjoint method).

**Bug 3: NumPy ComplexWarning Deprecation**
*   **Issue:** `train_gpu.py` failed on startup because it tried to filter `np.ComplexWarning`, which was removed in NumPy 2.x.
*   **Fix:** Commented out `# warnings.filterwarnings("ignore", category=np.ComplexWarning)` in `examples/demos/train_gpu.py`.

### 2. Training Execution
Ran `python examples/demos/train_gpu.py` using the CPU fallback.

**Model Details:**
*   **Backend:** `lightning.qubit` (C++ CPU, adjoint)
*   **Corpus:** 1359 chars, vocab size: 43
*   **Parameters:** 11,474 total (50 quantum)
*   **Epochs:** 5 (168 batches per epoch)

**Metrics (In Progress):**
*   **Epoch 1:** Loss = 3.0459 (Speed: ~2482ms/step, ~417.1s total per epoch)
*   *Note: Because it is running on the CPU instead of the raw GPU, it takes roughly 7 minutes per epoch (Total estimated time ~35 minutes).*
