# Antigravity Agent Directive: NVIDIA GPU Machine Instructions

> **Notice to the Antigravity Agent on this Machine:**  
> You are running on the **Secondary High-Performance Workstation (equipped with an NVIDIA GPU)** for the **Cognitive CanSat & TinyML Project**.  
> The primary laptop handles ground station UI, hardware serial links, and telemetry testing.  
> **Your primary mission on this machine is:** GPU-accelerated deep learning model training, hyperparameter optimization, INT8 model quantization, C++ flatbuffer generation, and syncing trained artifacts back to GitHub.

---

## 1. Golden Rules for this Machine

1. **Path Independence:** NEVER assume drive letters (like `E:` or `C:`). Always use relative paths from the repository root:
   ```python
   import os
   REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
   ```
2. **Git Workflow & Concurrency:**
   * Always run `git pull --rebase origin main` before starting new tasks to get the latest work pushed from Laptop 1.
   * Only commit and push files you produce: trained models in `ml/saved_models/`, `ml/model_metrics.json`, and C++ headers in `firmware/`.
   * NEVER force-push or remove files created by Laptop 1.
   * Verify that the 27,000 extracted image files in `data/eurosat/` remain ignored by `.gitignore`.

---

## 2. Step 1: Environment Verification & GPU Setup (Clean Machine Bootstrapping)

If this machine has **no Python, no virtualenv, and no coding tools installed**:

### Option A: Autonomous 1-Click Bootstrapper (Recommended)
Simply execute the included workstation bootstrapper:
```powershell
powershell -ExecutionPolicy Bypass -File .\setup_gpu_workstation.ps1
```
*(Or double-click `SETUP_GPU_WORKSTATION.bat` from File Explorer).*

**What this automatically executes with zero user intervention:**
1. Detects if Python is installed. If missing, silently downloads and installs official Python 3.11 64-bit and refreshes `PATH`.
2. Creates the local `.venv` virtual environment.
3. Installs CUDA-accelerated PyTorch (bundling CUDA 12.1 runtime - **no separate NVIDIA SDK download required!**).
4. Installs `requirements-ml.txt` (server deps plus ONNX, OpenCV and notebook tooling).
5. Extracts and verifies the 27,000 EuroSAT images from `data/EuroSAT_RGB.zip`.
6. Probes the GPU and outputs confirmation that Tensor Cores are active.

### Option B: Manual Setup (If Python is already installed)
```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### C. Install CUDA-Accelerated PyTorch & Dependencies
Run the official PyTorch CUDA 12.1 / 12.4 pip installer:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-ml.txt
```

### D. Verify GPU Acceleration (Sanity Check)
Run this command and ensure it reports `CUDA: True` with your GPU name:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

---

## 3. Step 2: Dataset Verification & Extraction

The 89.9 MB EuroSAT archive [`data/EuroSAT_RGB.zip`](file:///e:/Cansat/data/EuroSAT_RGB.zip) is already present in this repository.

Execute the dataset loader:
```powershell
python ml/download_dataset.py
```
* It will detect `data/EuroSAT_RGB.zip` on disk.
* It will extract all 27,000 images into `data/eurosat/2750/` in ~15 seconds without re-downloading anything.
* It verifies all 10 classes and confirms the dataset is ready.

---

## 4. Step 3: Train the TinyML Aerial Vision Model on GPU

Execute the dedicated training pipeline:
```powershell
python ml/train_tinyml_vision.py
```

### What this script executes:
1. Loads 27,000 Sentinel-2 images categorized into the 4 CanSat Safe Landing Area Index (SLAI) tiers:
   * **`SAFE_LZ`** (AnnualCrop, HerbaceousVegetation, Pasture, PermanentCrop)
   * **`OBSTACLE_CANOPY`** (Forest)
   * **`CRITICAL_HAZARD`** (Highway, Industrial, Residential)
   * **`WATER_HAZARD`** (River, SeaLake)
2. Automatically engages CUDA tensor cores with automatic mixed precision (`torch.amp.autocast('cuda')`).
3. Trains the ultra-compact `TinyLandingNet` architecture (~18k parameters, designed for < 30 KB flash footprint on ESP32-CAM).
4. Evaluates test set metrics (Target: **> 94% Macro F1-score**).
5. Outputs:
   * PyTorch weights: `ml/saved_models/tinyml_landing_safety.pth`
   * ONNX model: `ml/saved_models/tinyml_landing_safety.onnx`
   * Performance metrics & confusion matrix: `ml/model_metrics.json`

---

## 5. Step 4: Quantize to INT8 & Export C++ Header (`model_data.h`)

Run the INT8 quantization and C++ flatbuffer export script:
```powershell
python ml/export_tinyml_header.py
```
* Converts floating-point weights into 8-bit signed integer (`int8_t`) weight tables.
* Verifies total binary size is under **25 KB** (Achieved: **7.15 KB** / 7,320 parameters).
* Generates `firmware/esp32_cam_airborne/model_data.h` and `ml/saved_models/model_data.h`, ready to compile directly in the Arduino IDE for the ESP32-CAM.

---

## 6. Step 5: Benchmark Inference & Validate Embedded Constraints (Task ML-04)

Run the automated inference benchmarking pipeline:
```powershell
python ml/benchmark_inference.py
```
* Measures ONNX Runtime CPU latency (Achieved: **0.055 ms** / 18,073 FPS).
* Measures PyTorch CUDA GPU latency on RTX 4060 (Achieved: **0.614 ms** / 1,628 FPS).
* Computes edge microcontroller budget (AI-Thinker ESP32-CAM Xtensa LX6 @ 240 MHz):
  - Total MACs: **783,360** (1.567 MFLOPs)
  - Estimated Edge Latency: **~16.3 ms** (Target: < 150 ms per frame — **PASS**)
  - Flash ROM: **7.15 KB** (Target: < 25 KB — **PASS**)
* Validates full 3x3 spatial hazard grid segmentation and directional escape vector derivation on 320x240 QVGA frames in **4.23 ms** (236.6 FPS).
* Records all benchmark figures to `ml/model_metrics.json`.

---

## 7. Step 6: Sync Trained Artifacts Back to GitHub

Once training, quantization, and benchmarking are complete, sync the deliverables to GitHub:

```powershell
# 1. Pull latest upstream commits to avoid conflicts
git pull --rebase origin main

# 2. Stage only the newly generated model artifacts & code
git add ml/saved_models/ ml/model_metrics.json ml/benchmark_inference.py firmware/esp32_cam_airborne/model_data.h tasks.txt docs/

# 3. Verify that 27k image files in data/eurosat/ are NOT staged
git status

# 4. Commit and push
git commit -m "feat(ml): complete TinyLandingNet training, INT8 quantization, and inference benchmarking"
git push origin main
```

Once pushed, other workstations can run `git pull origin main` to instantly receive the trained models, benchmark telemetry, and compile the ESP32 firmware!

