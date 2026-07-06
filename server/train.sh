#!/usr/bin/env bash
# ============================================================
# YOLOv5 远程服务器 — 训练启动脚本
# ============================================================
# 在 Linux 服务器上执行:
#   bash server/train.sh --data data/datasets.yaml --epochs 200
#
# 支持 nohup 后台运行:
#   nohup bash server/train.sh --data data.yaml --epochs 200 > train.log 2>&1 &

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
YOLOV5_DIR="$PROJECT_ROOT/yolov5-7.0"
VENV_PYTHON="$PROJECT_ROOT/venv/bin/python"

# ============================================================
# 参数默认值
# ============================================================
DATA=""
WEIGHTS="yolov5n.pt"
EPOCHS=200
IMG_SIZE=320
BATCH_SIZE=0
WORKERS=0
DEVICE="0"
NAME=""
NO_AMP=false
NO_CACHE=false

# ============================================================
# 解析参数
# ============================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --data)       DATA="$2"; shift 2 ;;
        --weights)    WEIGHTS="$2"; shift 2 ;;
        --epochs)     EPOCHS="$2"; shift 2 ;;
        --imgsz)      IMG_SIZE="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --workers)    WORKERS="$2"; shift 2 ;;
        --device)     DEVICE="$2"; shift 2 ;;
        --name)       NAME="$2"; shift 2 ;;
        --no-amp)     NO_AMP=true; shift ;;
        --no-cache)   NO_CACHE=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================
# 验证必填参数
# ============================================================
if [ -z "$DATA" ]; then
    echo "[ERROR] --data is required"
    echo "Usage: bash server/train.sh --data <dataset.yaml> [options]"
    exit 1
fi

# ============================================================
# 环境检查
# ============================================================
if [ ! -f "$VENV_PYTHON" ]; then
    echo "[ERROR] Python venv not found at $VENV_PYTHON"
    echo "  Run first: bash server/setup.sh"
    exit 1
fi

cd "$YOLOV5_DIR"

# ============================================================
# 硬件检测 & 自适应参数
# ============================================================
echo ""
echo "========================================"
echo " YOLOv5 远程训练 — 硬件自适应配置"
echo "========================================"

CPU_CORES=$(nproc 2>/dev/null || echo 4)
RAM_GB=$(free -g 2>/dev/null | awk '/Mem:/ {print $2}' || echo 8)

HAS_GPU=false
GPU_NAME="N/A"
GPU_MEM=0

if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$GPU_INFO" ]; then
        HAS_GPU=true
        GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)
        GPU_MEM_MB=$(echo "$GPU_INFO" | cut -d',' -f2 | sed 's/ MiB//' | xargs)
        GPU_MEM=$(( GPU_MEM_MB / 1024 ))
    fi
fi

# 自适应 batch size
if [ "$BATCH_SIZE" -eq 0 ]; then
    if [ "$HAS_GPU" = true ]; then
        if [ "$GPU_MEM" -ge 24 ]; then
            BATCH_SIZE=128
        elif [ "$GPU_MEM" -ge 12 ]; then
            BATCH_SIZE=64
        elif [ "$GPU_MEM" -ge 8 ]; then
            BATCH_SIZE=32
        elif [ "$GPU_MEM" -ge 4 ]; then
            BATCH_SIZE=16
        else
            BATCH_SIZE=8
        fi
    else
        BATCH_SIZE=$(( RAM_GB / 4 ))
        [ "$BATCH_SIZE" -gt 8 ] && BATCH_SIZE=8
        [ "$BATCH_SIZE" -lt 2 ] && BATCH_SIZE=2
    fi
fi

# 自适应 workers
if [ "$WORKERS" -eq 0 ]; then
    WORKERS=$(( CPU_CORES / 2 ))
    [ "$WORKERS" -gt 8 ] && WORKERS=8
    [ "$WORKERS" -lt 2 ] && WORKERS=2
fi

# 实验名
if [ -z "$NAME" ]; then
    NAME="train_$(date +%Y%m%d_%H%M%S)"
fi

# 预训练权重准备
if [ "$WEIGHTS" = "yolov5n.pt" ] && [ ! -f "$YOLOV5_DIR/yolov5n.pt" ]; then
    echo "[DOWNLOAD] Downloading yolov5n.pt..."
    $VENV_PYTHON -c "from utils.downloads import attempt_download; attempt_download('yolov5n.pt')"
fi

echo "  服务器配置:"
echo "    CPU:       $CPU_CORES cores"
echo "    RAM:       ${RAM_GB} GB"
echo "    GPU:       $GPU_NAME ($GPU_MEM GB)"
echo ""
echo "  训练参数:"
echo "    数据集:    $DATA"
echo "    权重:      $WEIGHTS"
echo "    轮数:      $EPOCHS"
echo "    Batch:     $BATCH_SIZE"
echo "    尺寸:      $IMG_SIZE"
echo "    设备:      $DEVICE"
echo "    实验名:    $NAME"
echo "    Workers:   $WORKERS"
echo "    AMP:       $( [ "$NO_AMP" = false ] && echo 'ON' || echo 'OFF' )"
echo "========================================"
echo ""

# ============================================================
# 构建训练命令
# ============================================================
CMD=(
    "$VENV_PYTHON" train.py
    --data "$DATA"
    --weights "$WEIGHTS"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --imgsz "$IMG_SIZE"
    --device "$DEVICE"
    --workers "$WORKERS"
    --project runs/train
    --name "$NAME"
    --exist-ok
)

if [ "$NO_CACHE" = false ]; then
    CMD+=(--cache ram)
fi

echo "[CMD] ${CMD[*]}"
echo ""

# ============================================================
# 启动训练
# ============================================================
START_TIME=$(date +%s)

"${CMD[@]}"
EXIT_CODE=$?

END_TIME=$(date +%s)
DURATION=$(( (END_TIME - START_TIME) / 60 ))

# ============================================================
# 完成
# ============================================================
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "========================================"
    echo " 训练完成！"
    echo " 耗时: ${DURATION} 分钟"
    echo " 最佳模型: runs/train/$NAME/weights/best.pt"
    echo " 最后模型: runs/train/$NAME/weights/last.pt"
    echo " 指标文件: runs/train/$NAME/results.csv"
    echo "========================================"
else
    echo "[ERROR] 训练异常退出，退出码: $EXIT_CODE"
    echo "常见问题:"
    echo "  1. CUDA out of memory → 减小 --batch-size 或 --imgsz"
    echo "  2. 数据集路径错误 → 检查 $DATA 指向的目录是否存在"
    echo "  3. 依赖缺失 → 运行 bash server/setup.sh"
    exit $EXIT_CODE
fi
