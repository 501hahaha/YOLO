# YOLOv5 模型导出脚本 (ONNX/TensorRT/TorchScript)
# 用法: .\scripts\yolo\export.ps1 -Weights best.pt -Format onnx
param(
    [string]$Weights = "runs/train/exp/weights/best.pt",
    [string]$Format = "onnx",
    [int]$ImgSize = 640,
    [int]$Batch = 1,
    [switch]$Simplify
)

$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location "$root\yolov5-7.0"

$python = if (Test-Path "..\venv\Scripts\python.exe") { "..\venv\Scripts\python.exe" } else { "python" }

Write-Host "导出: $Weights -> $Format" -ForegroundColor Cyan

$args = @("export.py", "--weights", $Weights, "--imgsz", $ImgSize, "--batch", $Batch, "--include", $Format)
if ($Simplify) { $args += "--simplify" }

& $python $args
Write-Host "导出完成！" -ForegroundColor Green
