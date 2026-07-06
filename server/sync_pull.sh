#!/usr/bin/env bash
# ============================================================
# YOLOv5 远程服务器 — 结果拉取 (服务器 → 本地)
# ============================================================
# 在本地 Windows (Git Bash) 执行:
#   bash server/sync_pull.sh                     # 拉取最新训练结果
#   bash server/sync_pull.sh --name <exp_name>    # 拉取指定实验
#   bash server/sync_pull.sh --all                # 拉取全部 runs
#
# 拉取内容: best.pt, last.pt, results.csv, results.png, train.log, hyp.yaml, opt.yaml

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$SCRIPT_DIR/config.yaml"

# ============================================================
# 解析参数
# ============================================================
EXP_NAME=""
PULL_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --name) EXP_NAME="$2"; shift 2 ;;
        --all)  PULL_ALL=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================
# 解析 config.yaml
# ============================================================
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] Config not found: $CONFIG_FILE"
    exit 1
fi

parse_yaml() {
    python -c "
import yaml, sys
with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)
keys = '$1'.split('.')
val = cfg
for k in keys:
    val = val.get(k, {})
if isinstance(val, list):
    print(' '.join(val))
else:
    print(val or '')
" 2>/dev/null || {
        echo "[ERROR] Failed to parse config.yaml"
        exit 1
    }
}

HOST=$(parse_yaml "server.host")
PORT=$(parse_yaml "server.port")
USER=$(parse_yaml "server.user")
KEY_PATH=$(parse_yaml "server.key_path")
REMOTE_ROOT=$(parse_yaml "remote.project_root")

if [ -z "$HOST" ] || [ -z "$REMOTE_ROOT" ]; then
    echo "[ERROR] config.yaml missing required fields"
    exit 1
fi

PORT=${PORT:-22}

# ============================================================
# 构建 SSH/RSYNC 连接参数
# ============================================================
RSYNC_E="ssh -p $PORT"
if [ -n "$KEY_PATH" ]; then
    KEY_PATH_EXPANDED="${KEY_PATH/#\~/$HOME}"
    RSYNC_E="ssh -i $KEY_PATH_EXPANDED -p $PORT"
fi

SSH_PORT_OPT="-p $PORT"
SSH_KEY_OPT=""
if [ -n "$KEY_PATH" ]; then
    KEY_PATH_EXPANDED="${KEY_PATH/#\~/$HOME}"
    SSH_KEY_OPT="-i $KEY_PATH_EXPANDED"
fi

SSH_TARGET="${USER}@${HOST}"

# ============================================================
# 确定要拉取的实验
# ============================================================
RUNS_PATH="$REMOTE_ROOT/yolov5-7.0/runs/train"
LOCAL_RUNS="$PROJECT_ROOT/yolov5-7.0/runs/train"

echo ""
echo "========================================"
echo " 训练结果拉取 ← 远程服务器"
echo "========================================"
echo "  服务器:   $SSH_TARGET:$PORT"
echo "  远程:     $RUNS_PATH"
echo "  本地:     $LOCAL_RUNS"
echo "========================================"
echo ""

if [ "$PULL_ALL" = true ]; then
    # 拉取所有 runs
    echo "[RSYNC] Pulling all runs..."
    mkdir -p "$LOCAL_RUNS"
    rsync -avz --progress \
        -e "$RSYNC_E" \
        "$SSH_TARGET:$RUNS_PATH/" \
        "$LOCAL_RUNS/"
    echo ""
    echo "所有训练结果已拉取到: $LOCAL_RUNS"

elif [ -n "$EXP_NAME" ]; then
    # 拉取指定实验
    echo "[RSYNC] Pulling $EXP_NAME ..."
    mkdir -p "$LOCAL_RUNS/$EXP_NAME"
    rsync -avz --progress \
        -e "$RSYNC_E" \
        "$SSH_TARGET:$RUNS_PATH/$EXP_NAME/" \
        "$LOCAL_RUNS/$EXP_NAME/"
    echo ""
    echo "训练结果已拉取到: $LOCAL_RUNS/$EXP_NAME"

else
    # 拉取最新实验 (按修改时间排序)
    echo "[SSH] Finding latest run on server..."
    LATEST=$(ssh $SSH_PORT_OPT $SSH_KEY_OPT "$SSH_TARGET" \
        "ls -dt $RUNS_PATH/*/ 2>/dev/null | head -1" 2>/dev/null || echo "")

    if [ -z "$LATEST" ]; then
        echo "[ERROR] No runs found on server at $RUNS_PATH"
        exit 1
    fi

    LATEST_NAME=$(basename "$LATEST")
    echo "[INFO] Latest run: $LATEST_NAME"

    mkdir -p "$LOCAL_RUNS/$LATEST_NAME"
    rsync -avz --progress \
        -e "$RSYNC_E" \
        "$SSH_TARGET:$RUNS_PATH/$LATEST_NAME/" \
        "$LOCAL_RUNS/$LATEST_NAME/"

    echo ""
    echo "========================================"
    echo " 拉取完成！"
    echo "========================================"
    echo ""
    echo "结果位置: $LOCAL_RUNS/$LATEST_NAME"
    echo ""
    echo "关键文件:"
    ls -lh "$LOCAL_RUNS/$LATEST_NAME/weights/" 2>/dev/null || true
    echo ""
    echo "后续操作:"
    echo "  验证:  python yolov5-7.0/val.py --weights $LOCAL_RUNS/$LATEST_NAME/weights/best.pt --data <data.yaml>"
    echo "  导出:  python yolov5-7.0/export.py --weights $LOCAL_RUNS/$LATEST_NAME/weights/best.pt --include onnx"
    echo "  检测:  python yolov5-7.0/detect.py --weights $LOCAL_RUNS/$LATEST_NAME/weights/best.pt --source <image>"
fi
