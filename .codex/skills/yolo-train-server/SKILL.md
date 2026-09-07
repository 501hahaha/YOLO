---
name: yolo-train-server
description: YOLOv5 远程服务器训练 — SSH连接、代码同步、GPU训练、结果拉取。When the user wants to train a YOLOv5 model on a remote Linux GPU server (AutoDL, cloud GPU, etc.).
---

# YOLOv5 远程服务器训练

## 项目环境

- **框架**: YOLOv5 v7.0 (Ultralytics, GPL-3.0)
- **本地环境**: Windows (Git Bash)
- **服务器环境**: Linux + NVIDIA GPU (AutoDL / 自建服务器)
- **连接方式**: SSH + rsync
- **工程目录**: `server/`

## 前置条件

1. **服务器已开通**：已知 IP、端口、用户名、密码/密钥
2. **本地有 Git Bash**：能执行 bash 脚本
3. **本地有 rsync**：Git Bash 自带，或通过 `scoop install rsync`
4. **数据集已手动上传到服务器**：放在 `config.yaml` 中配置的 `data_root` 目录

## 快速开始

### Step 1: 配置服务器连接

编辑 `server/config.yaml`：

```bash
cd server
cp config.yaml config.yaml  # 已有模板，直接编辑
```

```yaml
server:
  host: "123.45.67.89"       # 服务器 IP
  port: 22                   # SSH 端口 (AutoDL 一般不是22)
  user: "root"               # 用户名
  # key_path: "~/.ssh/id_rsa" # 密钥登录 (可选)

remote:
  project_root: "/root/YOLO"
  data_root: "/root/YOLO/data"  # 数据集需手动上传到此目录
```

### Step 2: 首次 — 服务器环境安装

```bash
# 先推送脚本到服务器
bash server/sync_push.sh

# SSH 连接服务器执行安装 (5-10分钟)
bash server/connect.sh   # 连接后执行 ↓
cd /root/YOLO
bash server/setup.sh
```

### Step 3: 每次训练 — 三步走

```bash
# ① 推送代码到服务器
bash server/sync_push.sh

# ② 远程启动训练 (nohup 后台运行)
ssh -p 22 root@123.45.67.89 \
  "cd /root/YOLO && bash server/train.sh --data data/datasets.yaml --epochs 200 --imgsz 320"

# ③ 拉取训练结果
bash server/sync_pull.sh
```

### Step 4: 训练监控

```bash
# 快速 SSH 连接
bash server/connect.sh

# 在服务器上查看
tail -f runs/train/*/train.log      # 实时日志
nvidia-smi -l 1                      # 实时 GPU 使用
tensorboard --logdir runs/train/    # TensorBoard 可视化
```

## 命令模板

### sync_push — 推送代码

```bash
bash server/sync_push.sh
```

同步内容：
- `yolov5-7.0/` 核心代码 (排除 venv, __pycache__, runs, .git)
- `server/` 工程脚本

不同步：
- 数据集 (`data/` 图片) — 需手动上传一次
- 虚拟环境 (`venv/`)
- K230 部署相关 (`K230_Yolov5n/`)
- 标注工具 (`X-AnyLabeling/`)

### train — 远程训练

```bash
# 在服务器上执行 (通过 SSH)
bash server/train.sh \
  --data <数据集yaml> \
  --epochs <轮数> \
  --imgsz <尺寸> \
  --weights <预训练权重> \
  --batch-size <批次> \
  --name <实验名>
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data` | 数据集 yaml 路径 | 必填 |
| `--epochs` | 训练轮数 | `200` |
| `--imgsz` | 输入尺寸 | `320` |
| `--weights` | 预训练权重 | `yolov5n.pt` |
| `--batch-size` | 批次大小 | 自动检测 |
| `--name` | 实验名 | `train_YYYYMMDD_HHMMSS` |
| `--device` | GPU 设备 | `0` |
| `--workers` | 数据加载线程 | 自动检测 |
| `--no-amp` | 禁用混合精度 | 默认启用 |
| `--no-cache` | 不使用RAM缓存 | 默认启用cache |

### sync_pull — 拉取结果

```bash
# 拉取最新训练结果
bash server/sync_pull.sh

# 拉取指定实验
bash server/sync_pull.sh --name train_20240706_120000

# 拉取所有实验结果
bash server/sync_pull.sh --all
```

拉取内容：`best.pt`, `last.pt`, `results.csv`, `results.png`, `train.log`, `hyp.yaml`, `opt.yaml`

## 完整工作流示例

```bash
# === 首次使用 ===
# 1. 编辑配置
vim server/config.yaml

# 2. 推送代码
bash server/sync_push.sh

# 3. 连接服务器
bash server/connect.sh
# (手动上传数据集到 /root/YOLO/data/)
# bash server/setup.sh

# === 每次训练 ===
# 4. 本地修改代码后推送
bash server/sync_push.sh

# 5. 启动训练
ssh -p 2222 root@123.45.67.89 \
  "cd /root/YOLO && nohup bash server/train.sh --data data/datasets.yaml --epochs 200 --imgsz 320 > /tmp/train_nohup.log 2>&1 &"

# 6. 查看进度 (可选)
bash server/connect.sh
# tail -f runs/train/*/train.log

# 7. 训练结束后拉取
bash server/sync_pull.sh
```

## 训练输出

服务器端：
```
/root/YOLO/yolov5-7.0/runs/train/<exp名称>/
├── weights/
│   ├── last.pt       # 最后一轮权重
│   └── best.pt       # 最佳权重
├── results.csv       # 逐轮指标
├── results.png       # 指标曲线
├── train.log         # 训练日志 (nohup)
├── hyp.yaml          # 超参数
├── opt.yaml          # 训练配置
├── train_batch*.jpg  # 训练批次样例
└── val_batch*.jpg    # 验证批次样例
```

拉取到本地：
```
yolov5-7.0/runs/train/<exp名称>/   # 从服务器 rsync 拉取
```

## 关键参数（训练）

> 完整训练参数说明见 `yolo-train` skill。以下列出服务器特定默认值。

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `--data` | 数据集 yaml | `/root/YOLO/data/datasets.yaml` |
| `--weights` | 预训练权重 | `yolov5n.pt` |
| `--epochs` | 训练轮数 | `100~500` |
| `--imgsz` | 输入尺寸 | `320` (K230 适配) |
| `--batch-size` | 批次大小 | 根据 GPU 显存自动 |
| `--device` | GPU 设备号 | `0` |

## 训练完成后

拉取结果到本地后：
- 评估精度 → `yolo-validate` 技能（支持多模型并行对比）
- 一键到 K230 → `yolo-pipeline` 技能（train→validate→export→kmodel 全自动）

## 云平台适配

### AutoDL

```yaml
# server/config.yaml
server:
  host: "region-1.autodl.com"
  port: 22222                # AutoDL 分配的 SSH 端口
  user: "root"
  # key_path: "~/.ssh/id_rsa"

remote:
  project_root: "/root/YOLO"
  data_root: "/root/autodl-tmp/data"   # AutoDL 数据盘
```

AutoDL 特殊注意：
- SSH 端口不是默认 22，在"实例管理"页面查看
- 数据盘挂载在 `/root/autodl-tmp/`，建议数据集放此目录
- 系统盘容量小 (30GB)，不要在系统盘放大量数据

### 其他云平台

根据实际情况修改 `server/config.yaml` 中的 host/port/user 和远程路径即可。

## 与本地训练的对应关系

| 本地 (Windows) | 远程服务器 (Linux) |
|---|---|
| `train.ps1` | `server/train.sh` |
| `setup.ps1` | `server/setup.sh` |
| 手动复制文件 | `server/sync_push.sh` + `sync_pull.sh` |
| `yolo-train` skill | `yolo-train-server` skill |

## 常见问题

1. **rsync 找不到**: Git Bash 执行 `which rsync`，若没有则 `scoop install rsync` 或使用 MSYS2
2. **SSH 连接被拒**: 检查 IP/端口是否正确，AutoDL 端口一般不是 22
3. **服务器 CUDA 不可用**: `bash server/setup.sh` 会检测 GPU 并安装正确版本
4. **训练数据路径不对**: 检查服务器上 `data.yaml` 中的路径是否为远程路径
5. **nohup 进程意外退出**: 用 `tmux` 替代 nohup
   ```bash
   tmux new -s train
   bash server/train.sh --data data/datasets.yaml --epochs 200
   # Ctrl+B D 断开，tmux attach -t train 重新连接
   ```
6. **AutoDL 闲置关机**: 实例闲置一段时间会自动关机，训练中断。注意设置"无操作关机时间"。


