# ============================================================
# setup_windows.ps1  —  RTX 30xx (Ampere/sm_86) 向け FaceLift 環境
#   FaceLift 公式のテスト済み構成 (torch2.4 + cu124 + xformers) を使用。
#
#   事前に手動導入が必要:
#     - Miniconda
#     - Git
#     - Visual Studio 2022 + 「C++ によるデスクトップ開発」ワークロード
#       （rasterizer のビルドに cl.exe(MSVC) が必要）
#     - NVIDIA ドライバ
#
#   使い方（通常の PowerShell で可。cl.exe は本スクリプトが自動で用意します）:
#     git clone https://github.com/FuzukiKANNO/facelift-pipeline.git
#     cd facelift-pipeline
#     .\windows\setup_windows.ps1
#
#   ※ このファイルは UTF-8 (BOM 付き) で保存すること。
#      Windows PowerShell 5.1 は BOM 無し UTF-8 を ANSI 誤認し、日本語や
#      ヒアストリングが壊れてパースエラーになるため。
# ============================================================
$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot
$EnvName = if ($env:FACELIFT_ENV) { $env:FACELIFT_ENV } else { "facelift" }

function Banner($m){ Write-Host "`n==================================================" -ForegroundColor Cyan; Write-Host ">>> $m" -ForegroundColor Cyan; Write-Host "==================================================" -ForegroundColor Cyan }

# --- conda を PATH に用意（Miniconda を AddToPath 無しで入れていても動くように） ---
function Ensure-Conda {
  if (Get-Command conda -ErrorAction SilentlyContinue) { return }
  $cands = @(
    "$env:USERPROFILE\miniconda3", "$env:USERPROFILE\Anaconda3",
    "$env:LOCALAPPDATA\miniconda3", "C:\ProgramData\miniconda3", "C:\ProgramData\Anaconda3"
  )
  foreach ($c in $cands) {
    if (Test-Path "$c\Scripts\conda.exe") {
      $env:PATH = "$c;$c\Scripts;$c\Library\bin;$env:PATH"
      return
    }
  }
  throw "conda が見つかりません（Miniconda を導入してください）"
}

# --- cl.exe(MSVC) を PATH に用意（VS Developer Shell を自動で読み込む） ---
function Ensure-MSVC {
  if (Get-Command cl.exe -ErrorAction SilentlyContinue) { return }
  $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
  if (-not (Test-Path $vswhere)) { Write-Warning "vswhere が無く cl.exe を用意できません。段階5で失敗する可能性。"; return }
  $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
  if (-not $vsPath) { Write-Warning "C++ ツール付き VS が見つかりません。VS Build Tools の C++ ワークロードを入れてください。"; return }
  Import-Module "$vsPath\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
  Enter-VsDevShell -VsInstallPath $vsPath -DevCmdArguments "-arch=x64 -host_arch=x64" -SkipAutomaticLocation | Out-Null
  Set-Location $RepoRoot
}

Banner "0. 前提チェック"
Ensure-Conda
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { throw "nvidia-smi が見つかりません（NVIDIA ドライバ未導入）" }
nvidia-smi --query-gpu=name,memory.total --format=csv
Ensure-MSVC
if (Get-Command cl.exe -ErrorAction SilentlyContinue) { Write-Host "cl.exe OK: $((Get-Command cl.exe).Source)" }
else { Write-Warning "cl.exe が用意できませんでした。段階5(rasterizer ビルド)で失敗します。" }

Banner "1. conda 環境 ($EnvName, Python 3.10)"
try { conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>$null } catch {}
$envExists = (conda env list | Out-String) -match "(?m)^\s*$EnvName\s"
if (-not $envExists) {
  conda create -n $EnvName python=3.10 -c conda-forge --override-channels -y
} else {
  Write-Host "既存環境 '$EnvName' を使用"
}
conda run -n $EnvName python -m pip install --upgrade pip

Banner "2. PyTorch 2.4.0 (cu124, Ampere対応)"
conda run -n $EnvName pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
conda run -n $EnvName python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"

Banner "3. 依存パッケージ（公式構成 + xformers）"
conda run -n $EnvName pip install packaging==24.2 typing-extensions==4.14.0
conda run -n $EnvName pip install transformers==4.44.2 "diffusers[torch]==0.30.3" huggingface-hub==0.35.3 xformers==0.0.27.post2 accelerate==0.33.0
conda run -n $EnvName pip install Pillow==10.4.0 opencv-python==4.10.0.84 scikit-image==0.21.0 lpips==0.1.4
conda run -n $EnvName pip install facenet-pytorch --no-deps
conda run -n $EnvName pip install rembg onnxruntime
conda run -n $EnvName pip install numpy==1.26.4 matplotlib==3.7.5 scikit-learn==1.3.2 einops==0.8.0 jaxtyping==0.2.19 pytorch-msssim==1.0.0
conda run -n $EnvName pip install easydict==1.13 pyyaml==6.0.2 wandb==0.19.1 termcolor==2.4.0 plyfile==1.0.3 tqdm gradio==5.49.1
conda run -n $EnvName pip install videoio==0.3.0 ffmpeg-python==0.2.0 gdown
# ffmpeg 本体（ffprobe 含む）。--override-channels で defaults を使わず ToS 承認を回避。
conda install -n $EnvName -c conda-forge --override-channels ffmpeg -y

Banner "4. リポジトリ clone"
if (-not (Test-Path FaceLift)) { git clone https://github.com/weijielyu/FaceLift.git }
if (-not (Test-Path "face-parsing.PyTorch")) { git clone https://github.com/zllrunning/face-parsing.PyTorch.git }

Banner "5. diff-gaussian-rasterization ビルド (sm_86)"
conda install -n $EnvName -c nvidia -c conda-forge --override-channels cuda-toolkit=12.4 -y
New-Item -ItemType Directory -Force _build | Out-Null
$raster = "_build\diff-gaussian-rasterization"
if (-not (Test-Path $raster)) {
  git clone --recursive https://github.com/graphdeco-inria/diff-gaussian-rasterization.git $raster
} else {
  git -C $raster submodule update --init --recursive
}
# <cstdint> 追加パッチ（conda run の -c は複数行不可なのでファイルで実行）
conda run -n $EnvName python "windows\patch_rasterizer.py" $raster

$env:TORCH_CUDA_ARCH_LIST = "8.6"
$env:DISTUTILS_USE_SDK = "1"
# conda 版 CUDA は cudart.lib を <env>\Library\lib に置く（torch は \lib\x64 を探す）。
# LIB に追加してリンカが cudart.lib を見つけられるようにする。
$condaPrefix = (conda run -n $EnvName python -c "import sys; print(sys.prefix)").Trim()
$env:LIB = "$condaPrefix\Library\lib;$env:LIB"
Write-Host "LIB += $condaPrefix\Library\lib (cudart.lib present: $(Test-Path "$condaPrefix\Library\lib\cudart.lib"))"
conda run -n $EnvName pip install -v --no-build-isolation ".\$raster"
conda run -n $EnvName python -c "import diff_gaussian_rasterization; print('rasterizer import OK')"

Banner "6. BiSeNet 重み"
New-Item -ItemType Directory -Force "face-parsing.PyTorch\res\cp" | Out-Null
$w = "face-parsing.PyTorch\res\cp\79999_iter.pth"
if (-not (Test-Path $w)) { conda run -n $EnvName gdown "154JgKpzCPW82qINcVieuPH3fZ2e0P812" -O $w }
if (Test-Path $w) { Write-Host "重み OK: $((Get-Item $w).Length) bytes" }

Banner "セットアップ完了"
Write-Host "次: 顔画像を input\face.jpg に置き、  .\windows\run_windows.ps1  を実行"
