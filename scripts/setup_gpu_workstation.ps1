<#
.SYNOPSIS
    Automated Zero-to-Hero Environment Bootstrapper for NVIDIA GPU Laptop
    Cognitive CanSat & TinyML Vision Platform
.DESCRIPTION
    Runs on a completely fresh Windows machine with zero coding tools installed:
    1. Checks for Python. If missing, automatically downloads and installs Python 64-bit silently.
    2. Refreshes PATH in the active session without requiring a reboot.
    3. Creates local virtual environment (.venv).
    4. Installs CUDA-accelerated PyTorch (bundled with CUDA 12.1 runtime - no NVIDIA SDK needed).
    5. Installs project dependencies and ONNX.
    6. Extracts the local EuroSAT dataset archive.
    7. Validates GPU tensor acceleration.
#>

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "       COGNITIVE CANSAT - AUTOMATED NVIDIA GPU WORKSTATION BOOTSTRAPPER         " -ForegroundColor Cyan
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host ""

function Refresh-Path {
    $machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machinePath;$userPath"
}

# -----------------------------------------------------------------------------
# STEP 1: Verify or Automatically Install Python
# -----------------------------------------------------------------------------
Write-Host "[1/6] Checking for Python installation..." -ForegroundColor Yellow

$PythonCmd = Get-Command "python" -ErrorAction SilentlyContinue
if (-not $PythonCmd) {
    $PythonCmd = Get-Command "py" -ErrorAction SilentlyContinue
}

# Also check typical default installation directories
if (-not $PythonCmd) {
    $candidates = @(
        "$env:LocalAppData\Programs\Python\Python312\python.exe",
        "$env:LocalAppData\Programs\Python\Python311\python.exe",
        "$env:ProgramFiles\Python312\python.exe",
        "$env:ProgramFiles\Python311\python.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) {
            $env:Path = (Split-Path -Parent $c) + ";" + (Join-Path (Split-Path -Parent $c) "Scripts") + ";" + $env:Path
            $PythonCmd = Get-Command "python" -ErrorAction SilentlyContinue
            break
        }
    }
}

if (-not $PythonCmd) {
    Write-Host "[!] Python was not detected on this machine." -ForegroundColor Yellow
    Write-Host "[i] Automatically downloading and installing Python 3.11 (64-bit)..." -ForegroundColor Green

    $installedViaWinget = $false
    $winget = Get-Command "winget" -ErrorAction SilentlyContinue
    if ($winget) {
        try {
            Write-Host "    Attempting silent installation via Windows Package Manager (winget)..." -ForegroundColor Gray
            & winget install -e --id Python.Python.3.11 --scope currentUser --accept-package-agreements --accept-source-agreements --silent
            Refresh-Path
            $PythonCmd = Get-Command "python" -ErrorAction SilentlyContinue
            if ($PythonCmd) { $installedViaWinget = $true }
        } catch {
            Write-Host "    [!] Winget install attempt encountered an error. Falling back to direct installer..." -ForegroundColor Yellow
        }
    }

    if (-not $installedViaWinget) {
        $installerUrl = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
        $installerPath = Join-Path $env:TEMP "python-3.11.9-amd64.exe"
        
        Write-Host "    Downloading official installer from python.org..." -ForegroundColor Gray
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath -UseBasicParsing

        Write-Host "    Running silent installation (PrependPath=1, Include_pip=1)..." -ForegroundColor Gray
        $proc = Start-Process -FilePath $installerPath -ArgumentList "/quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_test=0 SimpleInstall=1" -Wait -PassThru
        
        Remove-Item $installerPath -Force -ErrorAction SilentlyContinue
        Refresh-Path
    }

    # Re-check paths
    $PythonCmd = Get-Command "python" -ErrorAction SilentlyContinue
    if (-not $PythonCmd) {
        $userPython = "$env:LocalAppData\Programs\Python\Python311\python.exe"
        if (Test-Path $userPython) {
            $env:Path = "$env:LocalAppData\Programs\Python\Python311;$env:LocalAppData\Programs\Python\Python311\Scripts;" + $env:Path
            $PythonCmd = Get-Command "python" -ErrorAction SilentlyContinue
        }
    }
}

if (-not $PythonCmd) {
    Write-Host "[FATAL] Automatic Python installation could not be completed." -ForegroundColor Red
    Write-Host "Please download and run the installer from https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "(Make sure to check 'Add python.exe to PATH' during installation)" -ForegroundColor Yellow
    exit 1
}

$pyVersion = & python --version 2>&1
Write-Host "[OK] Python is available: $pyVersion" -ForegroundColor Green

# -----------------------------------------------------------------------------
# STEP 2: Create Local Virtual Environment (.venv)
# -----------------------------------------------------------------------------
Write-Host "`n[2/6] Preparing virtual environment (.venv)..." -ForegroundColor Yellow
$VenvPath = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$VenvPip = Join-Path $VenvPath "Scripts\pip.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "    Creating new .venv directory..." -ForegroundColor Gray
    & python -m venv $VenvPath
}
Write-Host "[OK] Virtual environment ready at: $VenvPath" -ForegroundColor Green

# -----------------------------------------------------------------------------
# STEP 3: Install CUDA-Accelerated PyTorch (cu121)
# -----------------------------------------------------------------------------
Write-Host "`n[3/6] Installing CUDA-enabled PyTorch & TorchVision..." -ForegroundColor Yellow
Write-Host "    (Includes pre-packaged CUDA 12.1 runtime - no separate NVIDIA CUDA SDK required!)" -ForegroundColor Gray

& $VenvPip install --upgrade pip
& $VenvPip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
Write-Host "[OK] CUDA PyTorch installation complete." -ForegroundColor Green

# -----------------------------------------------------------------------------
# STEP 4: Install Project Dependencies & ONNX
# -----------------------------------------------------------------------------
Write-Host "`n[4/6] Installing project dependencies from requirements-ml.txt..." -ForegroundColor Yellow
& $VenvPip install -r (Join-Path $RepoRoot "requirements-ml.txt")
Write-Host "[OK] Dependencies installed successfully." -ForegroundColor Green

# -----------------------------------------------------------------------------
# STEP 5: Extract EuroSAT Aerial Dataset
# -----------------------------------------------------------------------------
Write-Host "`n[5/6] Verifying and unpacking EuroSAT aerial dataset..." -ForegroundColor Yellow
& $VenvPython (Join-Path $RepoRoot "ml\download_dataset.py")
Write-Host "[OK] Dataset verified on disk." -ForegroundColor Green

# -----------------------------------------------------------------------------
# STEP 6: Validate GPU Tensor Hardware
# -----------------------------------------------------------------------------
Write-Host "`n[6/6] Probing NVIDIA GPU hardware..." -ForegroundColor Yellow
$testGpuScript = @"
import torch
cuda_avail = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if cuda_avail else 'None'
count = torch.cuda.device_count() if cuda_avail else 0
print(f'CUDA_AVAILABLE:{cuda_avail}')
print(f'DEVICE_NAME:{device_name}')
print(f'DEVICE_COUNT:{count}')
"@

$gpuResult = & $VenvPython -c $testGpuScript
Write-Host $gpuResult -ForegroundColor Gray

if ($gpuResult -match "CUDA_AVAILABLE:True") {
    Write-Host "`n===============================================================================" -ForegroundColor Green
    Write-Host "  SUCCESS: NVIDIA GPU Acceleration is 100% ONLINE AND OPERATIONAL!            " -ForegroundColor Green
    Write-Host "===============================================================================" -ForegroundColor Green
    Write-Host "To train the vision model right now with full GPU acceleration, run:" -ForegroundColor Cyan
    Write-Host "  .\.venv\Scripts\python.exe ml\train_tinyml_vision.py" -ForegroundColor White
} else {
    Write-Host "`n[!] Notice: PyTorch installed successfully, but CUDA was not engaged." -ForegroundColor Yellow
    Write-Host "    Check that your laptop's NVIDIA graphics drivers are up to date." -ForegroundColor Yellow
    Write-Host "    Training will still execute cleanly using multi-core CPU mode." -ForegroundColor Gray
}
Write-Host ""
