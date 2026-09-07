# YOLOv5 检测脚本
# 用法: .\scripts\yolo\detect.ps1 -Source <图片/目录/视频/摄像头>
param(
    [string]$Source,
    [string]$Weights = "runs/train/exp/weights/best.pt",
    [float]$Conf = 0.25,
    [int]$ImgSize = 640,
    [string]$Device = "",
    [switch]$SaveTxt
)

$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location "$root\yolov5-7.0"

$python = if (Test-Path "..\venv\Scripts\python.exe") { "..\venv\Scripts\python.exe" } else { "python" }

if (-not $Source) {
    Write-Host "用法: .\scripts\yolo\detect.ps1 -Source <图片/目录>" -ForegroundColor Yellow
    Write-Host "示例: .\scripts\yolo\detect.ps1 -Source image.jpg" -ForegroundColor Gray
    Write-Host "示例: .\scripts\yolo\detect.ps1 -Source images/ -SaveTxt" -ForegroundColor Gray
    exit 1
}

$name = "detect_" + (Get-Date -Format "HHmmss")
$deviceArg = if ($Device) { $Device } else { "" }

Write-Host "检测: $Source -> runs/detect/$name" -ForegroundColor Cyan

$args = @("detect.py", "--weights", $Weights, "--source", $Source, "--imgsz", $ImgSize, "--conf-thres", $Conf, "--project", "runs/detect", "--name", $name, "--exist-ok")
if ($SaveTxt) { $args += "--save-txt"; $args += "--save-conf" }
if ($deviceArg) { $args += "--device"; $args += $deviceArg }

& $python $args
Write-Host "完成: runs/detect/$name" -ForegroundColor Green
