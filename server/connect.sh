#!/usr/bin/env bash
# ============================================================
# YOLOv5 远程服务器 — 快速 SSH 连接
# ============================================================
# 在本地 Windows (Git Bash) 执行:
#   bash server/connect.sh
#
# 读取 config.yaml 中的服务器信息，一键 SSH 连接

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.yaml"

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
val = cfg
for k in '$1'.split('.'):
    val = val.get(k, {})
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

if [ -z "$HOST" ]; then
    echo "[ERROR] config.yaml: server.host is required"
    exit 1
fi

PORT=${PORT:-22}

# ============================================================
# 构建 SSH 连接
# ============================================================
SSH_CMD="ssh"
if [ -n "$KEY_PATH" ]; then
    KEY_PATH_EXPANDED="${KEY_PATH/#\~/$HOME}"
    SSH_CMD="$SSH_CMD -i $KEY_PATH_EXPANDED"
fi
SSH_CMD="$SSH_CMD -p $PORT ${USER}@${HOST}"

echo ""
echo "连接服务器: ${USER}@${HOST}:${PORT}"
if [ -n "$REMOTE_ROOT" ]; then
    echo "工程目录:   $REMOTE_ROOT"
fi
echo ""

# 执行 SSH 连接
$SSH_CMD -t "cd ${REMOTE_ROOT} 2>/dev/null; exec bash -l"
