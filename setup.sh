#!/usr/bin/env bash
# ============================================================
# setup.sh  —  Linux + NVIDIA CUDA マシンで実行するセットアップ
# このリポジトリを Linux GPU マシンに転送してから実行してください。
#   bash setup.sh
# ============================================================
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE"
echo "[workspace] $WORKSPACE"

# ---- 前提チェック -------------------------------------------------
command -v conda >/dev/null 2>&1 || { echo "ERROR: conda が見つかりません"; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total --format=csv \
  || { echo "ERROR: NVIDIA GPU / nvidia-smi が見つかりません"; exit 1; }

# ---- 1. FaceLift ------------------------------------------------
if [ ! -d FaceLift ]; then
  git clone https://github.com/weijielyu/FaceLift.git
fi
cd FaceLift
echo "[FaceLift] setup_env.sh を実行します（conda 環境を作成）"
bash setup_env.sh
cd "$WORKSPACE"

echo "----------------------------------------------------------------"
echo "conda 環境一覧（FaceLift の環境名を確認してください）:"
conda env list
echo "----------------------------------------------------------------"

# 環境名は setup_env.sh 内の定義に合わせること（通常 facelift）
ENV_NAME="${FACELIFT_ENV:-facelift}"
echo "[env] 使用する conda 環境: $ENV_NAME  (変更する場合は FACELIFT_ENV=xxx bash setup.sh)"

# ---- 2. face-parsing.PyTorch (BiSeNet) --------------------------
if [ ! -d face-parsing.PyTorch ]; then
  git clone https://github.com/zllrunning/face-parsing.PyTorch.git
fi

# ---- 3. 追加の Python 依存 --------------------------------------
conda run -n "$ENV_NAME" pip install --quiet plyfile opencv-python pillow gdown

# ---- 4. BiSeNet 学習済み重み ------------------------------------
mkdir -p face-parsing.PyTorch/res/cp
WEIGHT=face-parsing.PyTorch/res/cp/79999_iter.pth
if [ ! -f "$WEIGHT" ]; then
  echo "[BiSeNet] 重みをダウンロード中..."
  conda run -n "$ENV_NAME" gdown "154JgKpzCPW82qINcVieuPH3fZ2e0P812" -O "$WEIGHT" \
    || echo "WARN: gdown 失敗。README の手動ダウンロード手順を参照してください。"
fi
[ -f "$WEIGHT" ] && echo "[BiSeNet] 重み OK: $WEIGHT"

echo ""
echo "[setup 完了] 次に input/face.jpg を配置し、run_pipeline.sh を実行してください。"
