# Quantum LLM Agent - Cross-Machine Run Log

This document tracks execution environments, bugs encountered, and performance metrics across different machines (e.g., MacBook vs. Ubuntu Desktop) to maintain shared knowledge across development environments.

## Run 1: Ubuntu Linux (NVIDIA GPU Setup)
**Date:** April 17, 2026
**Environment:** Ubuntu Linux, Python 3.12, CUDA 12
**Goal:** Run `setup_gpu.sh`, execute `examples/demos/train_gpu.py`, run tests, and run 12-qubit PoC.

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

**Metrics (Final):**
*   **Epoch 1 Loss:** 3.0459 (Speed: ~2482ms/step, ~417.1s total)
*   **Epoch 2 Loss:** 2.6801 (Speed: ~2449ms/step, ~411.4s total)
*   **Epoch 3 Loss:** 2.0502 (Speed: ~2427ms/step, ~407.7s total)
*   **Epoch 4 Loss:** 1.3416 (Speed: ~2468ms/step, ~414.6s total)
*   **Epoch 5 Loss:** 0.8595 (Speed: ~2450ms/step, ~411.6s total)
*   **Total Training Time:** 2062.3 seconds (~34.3 minutes)
*   **Final Output Check:** Generated a small sequence demonstrating the quantum transformer generation logic (`To beeee to e  or d m p surerereee reeperee e`). *Note: Coherence is low because it is a tiny 11K-parameter character-level toy model trained for 35 minutes on a small snippet of Shakespeare.*

### 3. Test Suite Execution
Ran `python -m pytest tests/ -v`.
*   **Result:** 51 passed (in ~2.84s).
*   **Details:** All unit tests, integration tests, quantum gradient flow tests, and memory tests successfully passed using the C++ adjoint simulation via `lightning.qubit`.

### 4. Proof-of-Concept Execution
Ran `python examples/demos/poc_12qubit.py`.
*   **Result:** Successfully ran the 12-qubit proof of concept logic end-to-end.
*   **Metrics:** 
    *   Forward pass (quantum attention + quantum embedding): 217.1 ms
    *   Auto-regressive generation (12 tokens): 638.1 ms
    *   Quantum reasoning module decision: 14.3 ms
    *   Quantum agent loop (2 steps): 7059.3 ms
*   **Architecture Verified:** Total params: 130,004. Total qubits: ~12. Verified that all components (Quantum attention, Quantum embedding, Quantum reasoning, Quantum memory) successfully communicate and process the agent loop.
