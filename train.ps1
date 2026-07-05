# YOLOv5 训练脚本 — 硬件自适应版
# 自动检测 GPU/CPU/RAM，选择最优训练参数
#
# 用法:
#   .\train.ps1 -Data data.yaml -Epochs 200
#   .\train.ps1 -Data data.yaml -Weights best.pt -Epochs 100   # 继续训练
#   .\train.ps1 -Data data.yaml -Epochs 300 -ImgSize 640       # 自定义尺寸
#
# 数据集要求:
#   data.yaml 需包含 path, train, val, nc, names 字段
#   支持 YOLO 格式标注 (.txt, class_id + normalized xywh)

param(
    [Parameter(Mandatory=$true)]
    [string]$Data,                              # 数据集 yaml 路径

    [string]$Weights = "yolov5n.pt",           # 预训练权重 (yolov5n.pt / best.pt)
    [int]$Epochs = 200,                        # 训练轮数
    [int]$ImgSize = 0,                         # 图片尺寸，0=自适应
    [int]$BatchSize = 0,                       # batch size，0=自适应
    [int]$Workers = 0,                         # 数据加载线程，0=自适应
    [string]$Device = "",                      # GPU设备号，"cpu"=只用CPU
    [string]$Name = "",                        # 实验名，空=自动生成
    [switch]$Resume,                           # 从断点恢复
    [switch]$NoCache                           # 不使用RAM缓存
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location "$root\yolov5-7.0"

# ============================================================
# 1. 检测 Python 环境
# ============================================================
$python = $null
if (Test-Path ".\venv\Scripts\python.exe") {
    $python = ".\venv\Scripts\python.exe"
} elseif (Test-Path "..\venv\Scripts\python.exe") {
    $python = "..\venv\Scripts\python.exe"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $python = "python"
} else {
    Write-Host "[ERROR] Python not found. Run: .\setup.ps1" -ForegroundColor Red
    exit 1
}

# 验证 PyTorch 是否可用
$torchCheck = & $python -c "import torch; print(torch.__version__)" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] PyTorch not installed. Run: .\setup.ps1" -ForegroundColor Red
    exit 1
}

# ============================================================
# 2. 硬件检测 & 自适应参数
# ============================================================
$hwInfo = & $python -c @"
import torch, psutil, os
cpu = psutil.cpu_count(logical=True)
ram = psutil.virtual_memory().total // (1024**3)
gpu_count = torch.cuda.device_count()
gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else 'N/A'
gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024**3) if gpu_count > 0 else 0
print(f'{cpu}|{ram}|{gpu_count}|{gpu_name}|{gpu_mem}')
"@
if ($LASTEXITCODE -ne 0) {
    # psutil not available, use defaults
    $cpuCount = [Environment]::ProcessorCount
    $ramGB = 8
    $gpuCount = 0
    $gpuName = "Unknown"
    $gpuMem = 0
} else {
    $parts = $hwInfo -split '\|'
    $cpuCount = [int]$parts[0]
    $ramGB = [int]$parts[1]
    $gpuCount = [int]$parts[2]
    $gpuName = $parts[3]
    $gpuMem = [int]$parts[4]
}

# 自适应 batch size
if ($BatchSize -eq 0) {
    if ($gpuCount -gt 0) {
        if ($gpuMem -ge 24)     { $BatchSize = 128 }
        elseif ($gpuMem -ge 12) { $BatchSize = 64 }
        elseif ($gpuMem -ge 8)  { $BatchSize = 32 }
        elseif ($gpuMem -ge 4)  { $BatchSize = 16 }
        else                    { $BatchSize = 8 }
    } else {
        # CPU only
        $BatchSize = [Math]::Min(8, [Math]::Max(2, $ramGB / 4))
    }
}

# 自适应图片尺寸
if ($ImgSize -eq 0) {
    $ImgSize = 640
}

# 自适应 workers
if ($Workers -eq 0) {
    $Workers = [Math]::Min(8, [Math]::Max(2, [int]($cpuCount / 2)))
}

# 自适应设备
$deviceArg = if ($Device -eq "cpu") { "cpu" } elseif ($Device -ne "") { $Device } elseif ($gpuCount -gt 0) { "0" } else { "cpu" }
$useAMP = ($deviceArg -ne "cpu")

# 缓存策略
$cacheArg = if ($NoCache) { "" } else { "--cache ram" }

if ($Name -eq "") {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $Name = "train_$timestamp"
}

# ============================================================
# 3. 显示配置 & 确认
# ============================================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " YOLOv5 训练 — 硬件自适应配置" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  主机配置:" -ForegroundColor White
Write-Host "    CPU:       $cpuCount 核" -ForegroundColor Gray
Write-Host "    RAM:       ${ramGB} GB" -ForegroundColor Gray
Write-Host "    GPU:       $gpuName ($gpuMem GB)" -ForegroundColor Gray
Write-Host ""
Write-Host "  训练参数:" -ForegroundColor White
Write-Host "    数据集:    $Data" -ForegroundColor Yellow
Write-Host "    权重:      $Weights" -ForegroundColor Yellow
Write-Host "    轮数:      $Epochs" -ForegroundColor Yellow
Write-Host "    Batch:     $BatchSize" -ForegroundColor Yellow
Write-Host "    尺寸:      $ImgSize" -ForegroundColor Yellow
Write-Host "    设备:      $deviceArg" -ForegroundColor Yellow
Write-Host "    AMP:       $useAMP" -ForegroundColor Yellow
Write-Host "    Workers:   $Workers" -ForegroundColor Yellow
Write-Host "    输出:      runs/train/$Name" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================
# 4. 下载预训练权重 (如需要)
# ============================================================
if ($Weights -eq "yolov5n.pt" -and -not (Test-Path "yolov5n.pt")) {
    Write-Host "[DOWNLOAD] 下载 yolov5n.pt..." -ForegroundColor Gray
    & $python -c "from utils.downloads import attempt_download; attempt_download('yolov5n.pt')"
}

# ============================================================
# 5. 训练
# ============================================================
$cmdArgs = @(
    "train.py",
    "--data", $Data,
    "--weights", $Weights,
    "--epochs", $Epochs,
    "--batch-size", $BatchSize,
    "--imgsz", $ImgSize,
    "--device", $deviceArg,
    "--workers", $Workers,
    "--project", "runs/train",
    "--name", $Name,
    "--exist-ok"
)

if ($cacheArg -ne "") { $cmdArgs += "--cache"; $cmdArgs += "ram" }
if ($Resume) { $cmdArgs += "--resume" }

Write-Host "[CMD] python $($cmdArgs -join ' ')" -ForegroundColor DarkGray
Write-Host ""

# 启动训练 (tqdm 进度条直接输出到终端)
$trainCmd = "& `$python $($cmdArgs -join ' ')"
Invoke-Expression $trainCmd
$exitCode = $LASTEXITCODE

# ============================================================
# 6. 完成
# ============================================================
Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host " 训练完成！" -ForegroundColor Green
    Write-Host " 最佳模型: runs/train/$Name/weights/best.pt" -ForegroundColor Green
    Write-Host " 最后模型: runs/train/$Name/weights/last.pt" -ForegroundColor Green
    Write-Host " 指标文件: runs/train/$Name/results.csv" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
} else {
    Write-Host "训练异常退出，退出码: $exitCode" -ForegroundColor Red
    Write-Host "常见问题:" -ForegroundColor Yellow
    Write-Host "  1. CUDA out of memory → 减小 batch-size 或 imgsz" -ForegroundColor Gray
    Write-Host "  2. 数据路径不存在 → 检查 $Data 指向的目录" -ForegroundColor Gray
    Write-Host "  3. 依赖缺失 → 运行 .\setup.ps1 安装环境" -ForegroundColor Gray
}
exit $exitCode
