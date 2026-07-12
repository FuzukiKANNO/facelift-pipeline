#!/usr/bin/env bash
# ============================================================
# run_pipeline.sh  —  FaceLift 推論 → Gaussian 分割 → 検証
#   bash run_pipeline.sh
# 事前に setup.sh を実行し、input/face.jpg を配置しておくこと。
# ============================================================
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE"
ENV_NAME="${FACELIFT_ENV:-facelift}"

# ---- 入力チェック ------------------------------------------------
[ -f input/face.jpg ] || { echo "ERROR: input/face.jpg がありません"; exit 1; }
mkdir -p facelift_input facelift_output segmented_output
cp -f input/face.jpg facelift_input/

# ---- 1. FaceLift 推論 -------------------------------------------
echo "[1/3] FaceLift 推論..."
cd FaceLift
conda run -n "$ENV_NAME" python inference.py \
  --input_dir ../facelift_input/ \
  --output_dir ../facelift_output/ \
  --seed 4 \
  --guidance_scale_2D 3.0 \
  --step_2D 50
cd "$WORKSPACE"

echo "[出力ファイル]"
find facelift_output/ -type f | sort

# ---- 2. .ply パス特定 -------------------------------------------
PLY_PATH="$(find facelift_output/ -name '*.ply' | head -1)"
[ -n "$PLY_PATH" ] || { echo "ERROR: facelift_output 内に .ply が見つかりません"; exit 1; }
echo "[使用する PLY] $PLY_PATH"

# ---- 3. Gaussian 分割 -------------------------------------------
echo "[2/3] Gaussian 分割..."
conda run -n "$ENV_NAME" python scripts/segment_gaussians.py \
  --ply_path "$PLY_PATH" \
  --face_image input/face.jpg \
  --face_parse_root face-parsing.PyTorch/ \
  --output_dir segmented_output/ \
  --device cuda

# ---- 4. 検証 ----------------------------------------------------
echo "[3/3] 検証..."
conda run -n "$ENV_NAME" python scripts/verify_ply.py \
  --output_dir segmented_output/ \
  --ply_path "$PLY_PATH"

echo ""
echo "[完了] segmented_output/ を確認し、debug_label_map.png で顔パース精度を目視確認してください。"
echo "投影がずれている場合は segment_gaussians.py に --flip_x / --no_flip_y を付けて再実行します。"
