# YOLOv5 → K230 kmodel 一键转换脚本
# 用法: .\scripts\k230\convert.ps1 -Model .\yolov5-7.0\runs\train\ball_320\weights\best.onnx -Dataset .\ball_image\datasets\yolo_dataset_10000\images\val
param(
    [Parameter(Mandatory=$true)]
    [string]$Model,                              # ONNX 模型路径

    [Parameter(Mandatory=$true)]
    [string]$Dataset,                            # 校准图片目录 (训练集里抽20张即可)

    [int]$InputWidth = 320,
    [int]$InputHeight = 320,
    [int]$PTQOption = 0,                        # 量化选项 0=uint8, 1=int16权重, 2=全int16
    [switch]$SkipVerify                         # 跳过精度验证
)

$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$toolsDir = Join-Path $root 'K230_Yolov5n\tools'
$resolvePath = {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path
    }
    $candidate = Join-Path $root $Path
    if (Test-Path -LiteralPath $candidate) {
        return (Resolve-Path -LiteralPath $candidate -ErrorAction Stop).Path
    }
    return $Path
}
$Model = & $resolvePath $Model
$Dataset = & $resolvePath $Dataset
Set-Location $toolsDir

$python = $(Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $python) { $python = "python" }

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " K230 kmodel 转换" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  模型:        $Model" -ForegroundColor Yellow
Write-Host "  校准集:      $Dataset" -ForegroundColor Yellow
Write-Host "  输入尺寸:    ${InputWidth}x${InputHeight}" -ForegroundColor Yellow
Write-Host "  量化选项:    $PTQOption" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: 转换 kmodel
Write-Host "[1/3] 转换 ONNX -> kmodel..." -ForegroundColor White
& $python to_kmodel.py `
    --target k230 `
    --model $Model `
    --dataset $Dataset `
    --input_width $InputWidth `
    --input_height $InputHeight `
    --ptq_option $PTQOption

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] kmodel 转换失败!" -ForegroundColor Red
    Write-Host "检查: 1) nncase 是否安装 2) dotnet-7 是否安装 3) 校准图片是否足够" -ForegroundColor Gray
    exit 1
}

# 自动推断生成的 kmodel 路径
$modelDir = Split-Path -Parent $Model
$modelBase = [System.IO.Path]::GetFileNameWithoutExtension($Model)
$kmodelPath = Join-Path $modelDir "$modelBase.kmodel"
Write-Host "[OK] kmodel -> $kmodelPath" -ForegroundColor Green

if ($SkipVerify) {
    Write-Host ""
    Write-Host "转换完成! kmodel: $kmodelPath" -ForegroundColor Green
    exit 0
}

# Step 2: 生成测试 bin 文件 (取校准集第一张图)
Write-Host "[2/3] 生成测试输入..." -ForegroundColor White
$sampleImg = Get-ChildItem $Dataset -Filter *.jpg | Select-Object -First 1
if (-not $sampleImg) { $sampleImg = Get-ChildItem $Dataset -Filter *.png | Select-Object -First 1 }
if (-not $sampleImg) {
    Write-Host "[WARN] 未找到测试图片，跳过验证" -ForegroundColor Yellow
    exit 0
}

$samplePath = $sampleImg.FullName
& $python save_bin.py `
    --image $samplePath `
    --save_path . `
    --input_width $InputWidth `
    --input_height $InputHeight

# Step 3: 精度验证 (余弦相似度)
Write-Host "[3/3] 精度验证 (ONNX vs kmodel)..." -ForegroundColor White
& $python simulate.py `
    --model $Model `
    --kmodel $kmodelPath `
    --model_input onnx_input_float32.bin `
    --kmodel_input kmodel_input_uint8.bin `
    --input_width $InputWidth `
    --input_height $InputHeight

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " 转换完成!" -ForegroundColor Green
Write-Host " kmodel: $kmodelPath" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "下一步: 将 $modelBase.kmodel 复制到 K230 的 /sdcard/mp_deployment_source/" -ForegroundColor Gray
Write-Host "         并修改 deploy_config.json 中的 kmodel_path" -ForegroundColor Gray
