# YOLOv5 模型验证脚本
# 用法: .\scripts\yolo\validate.ps1 -Weights best.pt -Data data.yaml
param(
    [string]$Weights = "runs/train/exp/weights/best.pt",
    [string]$Data = "data.yaml",
    [int]$ImgSize = 640,
    [switch]$Verbose
)

$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location "$root\yolov5-7.0"

$python = if (Test-Path "..\venv\Scripts\python.exe") { "..\venv\Scripts\python.exe" } else { "python" }

$name = "val_" + (Get-Date -Format "HHmmss")
Write-Host "验证: $Weights on $Data" -ForegroundColor Cyan

$args = @("val.py", "--weights", $Weights, "--data", $Data, "--imgsz", $ImgSize, "--project", "runs/val", "--name", $name, "--exist-ok")
if ($Verbose) { $args += "--verbose" }

& $python $args
Write-Host "完成: runs/val/$name" -ForegroundColor Green
