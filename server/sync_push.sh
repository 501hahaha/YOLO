#!/usr/bin/env bash
# ============================================================
# YOLOv5 远程服务器 — 代码同步 (本地 → 服务器)
# ============================================================
# 在本地 Windows (Git Bash) 执行:
#   bash server/sync_push.sh
#
# 同步内容:
#   - yolov5-7.0/ 核心代码
#   - server/ 工程脚本
#
# 不同步:
#   - venv/, __pycache__/, .git/, runs/
#   - data/, datasets/ (数据集手动上传)
#   - K230_Yolov5n/, X-AnyLabeling/
#   - *.kmodel, *.onnx, *.engine 等大文件

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$SCRIPT_DIR/config.yaml"

# ============================================================
# 解析 config.yaml
# ============================================================
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] Config not found: $CONFIG_FILE"
    echo "  Copy and edit config.yaml first"
    exit 1
fi

# 用 Python 解析 YAML (Git Bash 自带 python)
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
        echo "[ERROR] Failed to parse config.yaml. Is PyYAML installed?"
        echo "  pip install pyyaml"
        exit 1
    }
}

HOST=$(parse_yaml "server.host")
PORT=$(parse_yaml "server.port")
USER=$(parse_yaml "server.user")
KEY_PATH=$(parse_yaml "server.key_path")
REMOTE_ROOT=$(parse_yaml "remote.project_root")
SYNC_DIR=$(parse_yaml "local.sync_dir")
SCRIPTS_DIR=$(parse_yaml "local.scripts_dir")

if [ -z "$HOST" ] || [ -z "$REMOTE_ROOT" ]; then
    echo "[ERROR] config.yaml missing required fields (server.host, remote.project_root)"
    exit 1
fi

# ============================================================
# 构建 SSH/RSYNC 连接参数
# ============================================================
SSH_PORT_OPT=""
RSYNC_PORT_OPT=""
if [ "$PORT" != "22" ] && [ -n "$PORT" ]; then
    SSH_PORT_OPT="-p $PORT"
    RSYNC_PORT_OPT="-e 'ssh -p $PORT'"
fi

SSH_KEY_OPT=""
RSYNC_KEY_OPT=""
if [ -n "$KEY_PATH" ]; then
    # 展开 ~ 路径
    KEY_PATH_EXPANDED="${KEY_PATH/#\~/$HOME}"
    SSH_KEY_OPT="-i $KEY_PATH_EXPANDED"
    RSYNC_KEY_OPT="-e 'ssh -i $KEY_PATH_EXPANDED -p ${PORT:-22}'"
else
    RSYNC_KEY_OPT="-e 'ssh -p ${PORT:-22}'"
fi

SSH_TARGET="${USER}@${HOST}"
REMOTE_DEST="${SSH_TARGET}:${REMOTE_ROOT}"

# ============================================================
# 构建 exclude 参数
# ============================================================
EXCLUDE_ARGS=()
while IFS= read -r line; do
    line=$(echo "$line" | xargs)  # trim
    if [ -n "$line" ] && [[ ! "$line" =~ ^# ]]; then
        EXCLUDE_ARGS+=(--exclude="$line")
    fi
done < <(parse_yaml "sync.exclude" | tr ' ' '\n')

# ============================================================
# 显示配置
# ============================================================
echo ""
echo "========================================"
echo " 代码同步 → 远程服务器"
echo "========================================"
echo "  服务器:   $SSH_TARGET:$PORT"
echo "  远程目录: $REMOTE_ROOT"
echo "  同步目录: $SYNC_DIR + $SCRIPTS_DIR"
echo "========================================"
echo ""

# ============================================================
# 创建远程目录
# ============================================================
echo "[SSH] Creating remote directories..."
ssh $SSH_PORT_OPT $SSH_KEY_OPT "$SSH_TARGET" "mkdir -p $REMOTE_ROOT/$SYNC_DIR $REMOTE_ROOT/$SCRIPTS_DIR" 2>/dev/null || {
    echo "[WARN] Could not create remote directories (may already exist)"
}

# ============================================================
# 同步 yolov5-7.0/
# ============================================================
echo "[RSYNC] Syncing $SYNC_DIR/ ..."
rsync -avz --progress \
    $RSYNC_KEY_OPT \
    "${EXCLUDE_ARGS[@]}" \
    "$PROJECT_ROOT/$SYNC_DIR/" \
    "$REMOTE_DEST/$SYNC_DIR/"

echo ""

# ============================================================
# 同步 server/
# ============================================================
echo "[RSYNC] Syncing $SCRIPTS_DIR/ ..."
rsync -avz --progress \
    $RSYNC_KEY_OPT \
    --exclude="config.yaml" \
    "$PROJECT_ROOT/$SCRIPTS_DIR/" \
    "$REMOTE_DEST/$SCRIPTS_DIR/"

echo ""
echo "========================================"
echo " 同步完成！"
echo "========================================"
echo ""
echo "下一步:"
echo "  ssh $SSH_PORT_OPT $SSH_KEY_OPT $SSH_TARGET \"cd $REMOTE_ROOT && bash server/train.sh --data data/datasets.yaml --epochs 200\""
echo ""
