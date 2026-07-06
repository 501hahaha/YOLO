#!/usr/bin/env bash
# ============================================================
# YOLOv5 远程服务器 — 一键环境安装
# ============================================================
# 在 Linux 服务器上执行:
#   bash server/setup.sh
#
# 自动完成:
#   - 检测 Python
#   - 创建虚拟环境
#   - 检测 GPU → 安装 CUDA/CPU 版 PyTorch
#   - 安装 YOLOv5 依赖
#   - 验证安装

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
YOLOV5_DIR="$PROJECT_ROOT/yolov5-7.0"

echo ""
echo "========================================"
echo " YOLOv5 服务器环境安装"
echo "========================================"

# ============================================================
# 1. 查找 Python
# ============================================================
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        echo "[OK] Found: $cmd -> $($cmd --version 2>&1)"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "[ERROR] Python not found. Install Python 3.8+ first."
    echo "  Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
    echo "  CentOS/RHEL:   sudo yum install python3 python3-pip"
    echo "  AutoDL:        已预装 Python3"
    exit 1
fi

# ============================================================
# 2. 创建虚拟环境
# ============================================================
VENV_PATH="$PROJECT_ROOT/venv"

if [ ! -f "$VENV_PATH/bin/python" ]; then
    echo "[CREATE] Python virtual environment at: $VENV_PATH"
    $PYTHON -m venv "$VENV_PATH"
    echo "[OK] Virtual env created"
else
    echo "[OK] Virtual env already exists"
fi

PY="$VENV_PATH/bin/python"
PIP="$VENV_PATH/bin/pip"

# 升级 pip
$PIP install --upgrade pip -q

# ============================================================
# 3. 硬件检测 → 选择 PyTorch 版本
# ============================================================
echo ""
echo "--- 硬件检测 ---"

HAS_GPU=false
GPU_NAME=""
GPU_MEM=0

if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$GPU_INFO" ]; then
        HAS_GPU=true
        GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)
        GPU_MEM_MB=$(echo "$GPU_INFO" | cut -d',' -f2 | sed 's/ MiB//' | xargs)
        GPU_MEM=$(( GPU_MEM_MB / 1024 ))
        echo "[GPU] $GPU_NAME ($GPU_MEM GB)"
    fi
fi

if [ "$HAS_GPU" = false ]; then
    echo "[GPU] No NVIDIA GPU detected -> CPU mode"
fi

CPU_CORES=$(nproc 2>/dev/null || echo 4)
RAM_GB=$(free -g 2>/dev/null | awk '/Mem:/ {print $2}' || echo 8)
echo "[CPU] $CPU_CORES cores, RAM: ${RAM_GB}GB"

# ============================================================
# 4. 安装 PyTorch
# ============================================================
echo ""
echo "--- 安装依赖 ---"

if [ "$HAS_GPU" = true ]; then
    echo "[INSTALL] PyTorch (CUDA 12.4)..."
    # CUDA 12.4 适配大多数较新 GPU，AutoDL 等平台兼容
    $PIP install torch torchvision --index-url https://download.pytorch.org/whl/cu124
else
    echo "[INSTALL] PyTorch (CPU only)..."
    $PIP install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

if [ $? -ne 0 ]; then
    echo "[WARN] PyTorch CUDA 12.4 failed, trying CUDA 11.8..."
    $PIP install torch torchvision --index-url https://download.pytorch.org/whl/cu118
fi

# ============================================================
# 5. 安装 YOLOv5 依赖
# ============================================================
echo "[INSTALL] YOLOv5 requirements..."
$PIP install -r "$YOLOV5_DIR/requirements.txt"

# Pillow 版本锁定 (YOLOv5 v7.0 兼容)
echo "[FIX] Pinning Pillow==9.5..."
$PIP install "Pillow==9.5"

# ============================================================
# 6. 验证安装
# ============================================================
echo ""
echo "--- 验证安装 ---"

$PY -c "
import torch, sys
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:      {torch.cuda.get_device_name(0)}')
    vram_gb = torch.cuda.get_device_properties(0).total_memory // 1024**3
    print(f'VRAM:     {vram_gb} GB')
print(f'Python:   {sys.version}')
print('OK - Environment ready!')
"

# ============================================================
# 7. 完成
# ============================================================
echo ""
echo "========================================"
echo " 环境安装完成！"
echo "========================================"
echo ""
echo "接下来:"
echo "  1. 确保数据集已上传至服务器 data/ 目录"
echo "  2. 修改 data.yaml 中的 path 指向服务器路径"
echo "  3. 开始训练: bash server/train.sh --data data/datasets.yaml"
echo ""
