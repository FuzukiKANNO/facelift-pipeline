# ============================================================
# run_windows.ps1  —  FaceLift 推論 → パーツ分割 → 検証（Windows）
#   使い方（顔画像を input\face.jpg に置いてから）:
#     .\windows\run_windows.ps1
#   再推論を強制: $env:FORCE_INFER="1" にしてから実行
#
#   ※ このファイルは UTF-8 (BOM 付き) で保存すること（PowerShell 5.1 対策）。
# ============================================================
$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot
$EnvName = if ($env:FACELIFT_ENV) { $env:FACELIFT_ENV } else { "facelift" }

# conda を PATH に用意（AddToPath 無しの Miniconda でも動くように）
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
  $cands = @("$env:USERPROFILE\miniconda3","$env:USERPROFILE\Anaconda3","$env:LOCALAPPDATA\miniconda3","C:\ProgramData\miniconda3","C:\ProgramData\Anaconda3")
  foreach ($c in $cands) { if (Test-Path "$c\Scripts\conda.exe") { $env:PATH = "$c;$c\Scripts;$c\Library\bin;$env:PATH"; break } }
}
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) { throw "conda が見つかりません（Miniconda を導入してください）" }

if (-not (Test-Path "input\face.jpg")) { throw "input\face.jpg がありません。顔画像を配置してください。" }
New-Item -ItemType Directory -Force facelift_input, facelift_output, segmented_output | Out-Null
Copy-Item -Force "input\face.jpg" "facelift_input\"

# メモリ断片化対策（8GB VRAM なので OOM 回避を試みる）
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
# Python 標準出力を UTF-8 に（日本語 Windows の既定 cp932 だと出力文字で UnicodeEncodeError になるため）
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "[1/3] FaceLift 推論..." -ForegroundColor Cyan
$existing = Get-ChildItem -Recurse facelift_output -Filter *.ply -ErrorAction SilentlyContinue | Select-Object -First 1
if ($existing -and $env:FORCE_INFER -ne "1") {
  Write-Host "既存 .ply を再利用（推論スキップ）: $($existing.FullName)  （再推論は FORCE_INFER=1）"
} else {
  Push-Location FaceLift
  conda run -n $EnvName --no-capture-output python inference.py -i ../facelift_input/ -o ../facelift_output/ --seed 4 --guidance_scale_2D 3.0 --step_2D 50
  Pop-Location
}

$ply = Get-ChildItem -Recurse facelift_output -Filter *.ply | Select-Object -First 1
if (-not $ply) { throw "facelift_output に .ply が見つかりません" }
Write-Host "使用 PLY: $($ply.FullName)"

Write-Host "[2/3] パーツ分割..." -ForegroundColor Cyan
# BiSeNet は FaceLift がクロップした input.png（Gaussian が対応する前面像）にかける。
# 投影は簡易 bbox ではなく FaceLift の実カメラ(opencv_cameras.json frame=2 前面)を使う
# → パーツのスケール・位置が顔画像と正確に一致する。
$faceImg = Join-Path $ply.DirectoryName "input.png"
if (-not (Test-Path $faceImg)) { $faceImg = "input/face.jpg" }  # フォールバック
$camJson = "FaceLift/utils_folder/opencv_cameras.json"
conda run -n $EnvName --no-capture-output python scripts/segment_gaussians.py `
  --ply_path $ply.FullName `
  --face_image $faceImg `
  --face_parse_root "face-parsing.PyTorch/" `
  --output_dir "segmented_output/" `
  --camera_json $camJson `
  --camera_index 2 `
  --device cuda

Write-Host "[3/3] 検証..." -ForegroundColor Cyan
conda run -n $EnvName --no-capture-output python scripts/verify_ply.py --output_dir "segmented_output/" --ply_path $ply.FullName

Write-Host "`n完了。segmented_output\ を確認してください。" -ForegroundColor Green
