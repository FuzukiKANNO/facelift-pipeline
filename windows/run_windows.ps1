# ============================================================
# run_windows.ps1  —  FaceLift 推論 → パーツ分割 → 検証（Windows）
#   使い方（顔画像を input\face.jpg に置いてから）:
#     .\windows\run_windows.ps1
# ============================================================
$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot
$EnvName = if ($env:FACELIFT_ENV) { $env:FACELIFT_ENV } else { "facelift" }

if (-not (Test-Path "input\face.jpg")) { throw "input\face.jpg がありません。顔画像を配置してください。" }
New-Item -ItemType Directory -Force facelift_input, facelift_output, segmented_output | Out-Null
Copy-Item -Force "input\face.jpg" "facelift_input\"

# メモリ断片化対策（8GB VRAM なので OOM 回避を試みる）
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

Write-Host "[1/3] FaceLift 推論..." -ForegroundColor Cyan
$existing = Get-ChildItem -Recurse facelift_output -Filter *.ply -ErrorAction SilentlyContinue | Select-Object -First 1
if ($existing -and $env:FORCE_INFER -ne "1") {
  Write-Host "既存 .ply を再利用（推論スキップ）: $($existing.FullName)  （再推論は FORCE_INFER=1）"
} else {
  Push-Location FaceLift
  conda run -n $EnvName python inference.py -i ../facelift_input/ -o ../facelift_output/ --seed 4 --guidance_scale_2D 3.0 --step_2D 50
  Pop-Location
}

$ply = Get-ChildItem -Recurse facelift_output -Filter *.ply | Select-Object -First 1
if (-not $ply) { throw "facelift_output に .ply が見つかりません" }
Write-Host "使用 PLY: $($ply.FullName)"

Write-Host "[2/3] パーツ分割..." -ForegroundColor Cyan
conda run -n $EnvName python scripts/segment_gaussians.py `
  --ply_path $ply.FullName `
  --face_image "input/face.jpg" `
  --face_parse_root "face-parsing.PyTorch/" `
  --output_dir "segmented_output/" `
  --device cuda

Write-Host "[3/3] 検証..." -ForegroundColor Cyan
conda run -n $EnvName python scripts/verify_ply.py --output_dir "segmented_output/" --ply_path $ply.FullName

Write-Host "`n完了。segmented_output\ を確認してください。" -ForegroundColor Green
