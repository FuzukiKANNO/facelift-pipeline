#!/usr/bin/env bash
# ============================================================
# setup_blackwell.sh
#   RTX 5090 (Blackwell / sm_120) 向けセットアップ。
#   FaceLift 純正 setup_env.sh の torch==2.4.0+cu124 は Blackwell 非対応のため、
#   torch を cu128 系に差し替えた版。
#
#   使い方（Linux GPU マシン、conda 導入済み）:
#     cd ~/facelift-pipeline
#     git pull
#     bash setup_blackwell.sh
#
#   環境名を変えたい場合:  FACELIFT_ENV=myenv bash setup_blackwell.sh
# ============================================================
set -uo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE"
ENV_NAME="${FACELIFT_ENV:-facelift}"

banner() { echo ""; echo "=================================================="; echo ">>> $1"; echo "=================================================="; }
die()    { echo "ERROR: $1" >&2; exit 1; }

# ---- 前提チェック ------------------------------------------------
banner "0. 前提チェック"
command -v conda >/dev/null 2>&1 || die "conda が見つかりません"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi が見つかりません（GPUドライバ未導入）"
nvidia-smi --query-gpu=name,memory.total --format=csv

# conda をスクリプト内で activate できるようにする
source "$(conda info --base)/etc/profile.d/conda.sh"

# ---- 1. conda 環境（Python 3.10）--------------------------------
banner "1. conda 環境 '$ENV_NAME' (Python 3.10)"
# 新しい conda では defaults チャンネルの ToS 未承認で create が失敗するため、
# 念のため承認を試みる（失敗しても無視）。環境自体は conda-forge から作る。
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    2>/dev/null || true
if conda env list | grep -qE "^${ENV_NAME}\s"; then
  echo "既存の環境 '$ENV_NAME' を使用します"
else
  # conda-forge から作成（defaults チャンネルの ToS 問題を回避）
  conda create -n "$ENV_NAME" python=3.10 -c conda-forge --override-channels -y \
    || die "conda create 失敗（上の conda 出力を確認してください）"
fi
conda activate "$ENV_NAME" || die "conda activate 失敗"
echo "active python: $(which python)"
python -m pip install --upgrade pip

# ---- 2. Blackwell 対応 PyTorch (cu128) --------------------------
banner "2. PyTorch (cu128, Blackwell対応)"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 \
  || die "torch(cu128) のインストール失敗"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))  # RTX 5090 -> (12, 0)
PY

# ---- 3. 依存パッケージ（torch の pin は流用しない）---------------
banner "3. 依存パッケージ"
pip install packaging==24.2 typing-extensions==4.14.0
pip install transformers==4.44.2 "diffusers[torch]==0.30.3" huggingface-hub==0.35.3 accelerate==0.33.0
pip install Pillow==10.4.0 opencv-python==4.10.0.84 scikit-image==0.21.0 lpips==0.1.4
pip install facenet-pytorch --no-deps
pip install rembg
pip install numpy==1.26.4 matplotlib==3.7.5 scikit-learn==1.3.2 einops==0.8.0 jaxtyping==0.2.19 pytorch-msssim==1.0.0
pip install easydict==1.13 pyyaml==6.0.2 wandb==0.19.1 termcolor==2.4.0 plyfile==1.0.3 tqdm gradio==5.49.1
pip install videoio==0.3.0 ffmpeg-python==0.2.0
# 分割側で使用
pip install gdown

# ffmpeg（sudo が必要。パスワードを聞かれたら入力してください）
banner "3b. ffmpeg (apt, sudo)"
if command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg は既に導入済み"
else
  sudo apt update && sudo apt install -y ffmpeg || echo "WARN: ffmpeg 導入失敗（後で手動導入可）"
fi

# ---- 4. CUDA Toolkit 12.8 + rasterizer ビルド -------------------
banner "4. diff-gaussian-rasterization を sm_120 向けにビルド"
conda install -c nvidia -c conda-forge --override-channels cuda-toolkit=12.8 -y \
  || die "cuda-toolkit=12.8 の導入失敗"
export TORCH_CUDA_ARCH_LIST="12.0"
echo "nvcc: $(which nvcc)  / TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
nvcc --version | tail -2 || true
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization \
  || die "diff-gaussian-rasterization のビルド失敗（上のエラーを確認）"

# ---- 5. リポジトリ clone（未取得なら）---------------------------
banner "5. FaceLift / face-parsing.PyTorch"
[ -d FaceLift ]              || git clone https://github.com/weijielyu/FaceLift.git
[ -d face-parsing.PyTorch ]  || git clone https://github.com/zllrunning/face-parsing.PyTorch.git

# ---- 6. BiSeNet 重み --------------------------------------------
banner "6. BiSeNet 重み"
mkdir -p face-parsing.PyTorch/res/cp
WEIGHT=face-parsing.PyTorch/res/cp/79999_iter.pth
if [ ! -f "$WEIGHT" ]; then
  gdown "154JgKpzCPW82qINcVieuPH3fZ2e0P812" -O "$WEIGHT" \
    || echo "WARN: 重みDL失敗。README の手動DL手順を参照。"
fi
[ -f "$WEIGHT" ] && echo "重み OK: $WEIGHT"

banner "セットアップ完了"
echo "次の手順:"
echo "  1) 顔画像を配置:  cp /path/to/face.jpg input/face.jpg"
echo "  2) 実行:          FACELIFT_ENV=$ENV_NAME bash run_pipeline.sh"
