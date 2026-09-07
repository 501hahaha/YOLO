# YOLOv5 环境安装脚本
# 用法: .\scripts\yolo\setup.ps1
# 自动检测硬件，创建虚拟环境并安装 PyTorch + 依赖

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $root

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " YOLOv5 环境安装" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# ============================================================
# 1. 查找系统 Python
# ============================================================
$pythonBase = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $v = & $cmd --version 2>&1
        $pythonBase = $cmd
        Write-Host "[OK] Found: $cmd -> $v" -ForegroundColor Green
        break
    } catch {}
}
if (-not $pythonBase) {
    Write-Host "[ERROR] Python not found. Please install Python 3.8+ from https://python.org" -ForegroundColor Red
    exit 1
}

# ============================================================
# 2. 创建虚拟环境
# ============================================================
$venvPath = "$root\venv"
if (-not (Test-Path "$venvPath\Scripts\python.exe")) {
    Write-Host "[CREATE] Creating Python virtual environment..." -ForegroundColor Yellow
    & $pythonBase -m venv $venvPath
    Write-Host "[OK] Virtual env created at: venv/" -ForegroundColor Green
} else {
    Write-Host "[OK] Virtual env already exists" -ForegroundColor Green
}

# 激活虚拟环境
$py = "$venvPath\Scripts\python.exe"
$pip = "$venvPath\Scripts\pip.exe"

# ============================================================
# 3. 硬件检测 → 选 PyTorch 版本
# ============================================================
Write-Host ""
Write-Host "--- 检测硬件 ---" -ForegroundColor Cyan

# 检测 NVIDIA GPU
$hasGPU = $false
$gpuName = ""
try {
    $nvidiaCheck = & nvidia-smi --query-gpu=name --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0) {
        $hasGPU = $true
        $gpuName = $nvidiaCheck.Trim()
        Write-Host "[GPU] $gpuName" -ForegroundColor Green
    }
} catch {
    $hasGPU = $false
    Write-Host "[GPU] No NVIDIA GPU detected" -ForegroundColor Yellow
}

# 检测 CPU 架构 & RAM
$cpuCores = [Environment]::ProcessorCount
$ramGB = [Math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
Write-Host "[CPU] $cpuCores cores, RAM: ${ramGB}GB" -ForegroundColor Gray

# ============================================================
# 4. 升级 pip & 安装 PyTorch
# ============================================================
Write-Host ""
Write-Host "--- 安装依赖 ---" -ForegroundColor Cyan

& $pip install --upgrade pip -q

if ($hasGPU) {
    Write-Host "[INSTALL] PyTorch (CUDA)..." -ForegroundColor Yellow
    & $pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
} else {
    Write-Host "[INSTALL] PyTorch (CPU only)..." -ForegroundColor Yellow
    & $pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] PyTorch install failed. Trying default pip..." -ForegroundColor Yellow
    & $pip install torch torchvision
}

# ============================================================
# 5. 安装 YOLOv5 依赖
# ============================================================
Write-Host "[INSTALL] YOLOv5 requirements..." -ForegroundColor Yellow
& $pip install -r yolov5-7.0\requirements.txt

# Pillow 版本限制 (YOLOv5 v7 兼容)
Write-Host "[FIX] Pinning Pillow==9.5 for compatibility..." -ForegroundColor Gray
& $pip install "Pillow==9.5"

# ============================================================
# 6. 验证安装
# ============================================================
Write-Host ""
Write-Host "--- 验证安装 ---" -ForegroundColor Cyan

& $py -c @"
import torch, sys
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:      {torch.cuda.get_device_name(0)}')
    print(f'VRAM:     {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB')
print(f'Python:   {sys.version}')
print('OK - Environment ready!')
"@

# ============================================================
# 7. 完成
# ============================================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " 环境安装完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "接下来:" -ForegroundColor White
Write-Host "  1. 准备数据集 (参考 docs\YOLO_TRAINING.md)" -ForegroundColor Gray
Write-Host "  2. 开始训练: .\scripts\yolo\train.ps1 -Data your_data.yaml" -ForegroundColor Gray
Write-Host "  3. 检测:     .\scripts\yolo\detect.ps1 -Source image.jpg" -ForegroundColor Gray
Write-Host ""
