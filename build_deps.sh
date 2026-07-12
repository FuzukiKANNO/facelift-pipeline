#!/usr/bin/env bash
# ============================================================
# build_deps.sh
#   setup_blackwell.sh の段階4以降でこけた分を個別に立て直す:
#     1. diff-gaussian-rasterization を submodule(glm)込みでビルド
#     2. face-parsing.PyTorch を clone
#     3. BiSeNet 重みを取得（gdown --fuzzy フォールバック付き）
#
#   使い方:
#     cd ~/facelift-pipeline && git pull && bash build_deps.sh
# ============================================================
set -o pipefail   # set -u は使わない（conda activate 対策）
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE"
ENV_NAME="${FACELIFT_ENV:-facelift}"

banner() { echo ""; echo "=================================================="; echo ">>> $1"; echo "=================================================="; }

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME" || { echo "NG: conda 環境 '$ENV_NAME' を activate できません"; exit 1; }
echo "python: $(which python)"

# ---- 1. diff-gaussian-rasterization ----------------------------
banner "1. diff-gaussian-rasterization (glm submodule 込みでビルド)"
pip install ninja >/dev/null 2>&1
mkdir -p _build
RASTER_DIR=_build/diff-gaussian-rasterization
if [ ! -d "$RASTER_DIR" ]; then
  git clone --recursive https://github.com/graphdeco-inria/diff-gaussian-rasterization.git "$RASTER_DIR"
else
  echo "既存 clone を使用（submodule 更新）"
  git -C "$RASTER_DIR" submodule update --init --recursive
fi
# glm ヘッダの存在確認
if [ ! -d "$RASTER_DIR/third_party/glm/glm" ]; then
  echo "WARN: glm submodule が空です。再取得を試みます。"
  git -C "$RASTER_DIR" submodule update --init --recursive
fi

# RTX 5090 = sm_120 向けにビルド。詳細ログを出す（失敗時の原因特定用）
export TORCH_CUDA_ARCH_LIST="12.0"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST  nvcc=$(which nvcc)"
pip install -v "./$RASTER_DIR" 2>&1 | tee _build/raster_build.log
BUILD_RC=${PIPESTATUS[0]}

if [ "$BUILD_RC" -ne 0 ]; then
  echo ""
  echo "########################################################"
  echo "# rasterizer ビルド失敗 (rc=$BUILD_RC)"
  echo "# ログ: _build/raster_build.log の末尾を確認してください"
  echo "########################################################"
  tail -30 _build/raster_build.log
  echo ">>> このログ(特に error: の行)をスクショで共有してください。"
  # ここで止めず、face-parsing / 重みは続行する
fi

# ---- 2. face-parsing.PyTorch -----------------------------------
banner "2. face-parsing.PyTorch clone"
[ -d face-parsing.PyTorch ] || git clone https://github.com/zllrunning/face-parsing.PyTorch.git
ls -d face-parsing.PyTorch && echo "OK" || echo "NG"

# ---- 3. BiSeNet 重み -------------------------------------------
banner "3. BiSeNet 重み取得"
pip install -U gdown >/dev/null 2>&1
mkdir -p face-parsing.PyTorch/res/cp
WEIGHT=face-parsing.PyTorch/res/cp/79999_iter.pth
FILE_ID=154JgKpzCPW82qINcVieuPH3fZ2e0P812
if [ -f "$WEIGHT" ] && [ "$(stat -c%s "$WEIGHT" 2>/dev/null || echo 0)" -gt 1000000 ]; then
  echo "重み既に存在: $(du -h "$WEIGHT" | cut -f1)"
else
  echo "gdown --fuzzy で取得を試みます..."
  gdown --fuzzy "https://drive.google.com/uc?id=${FILE_ID}" -O "$WEIGHT" || \
  gdown "$FILE_ID" -O "$WEIGHT" || \
  echo "WARN: 自動DL失敗。ブラウザで下記から手動DLし $WEIGHT に置いてください:
    https://drive.google.com/file/d/${FILE_ID}/view"
fi
if [ -f "$WEIGHT" ]; then
  SZ=$(stat -c%s "$WEIGHT" 2>/dev/null || echo 0)
  echo "重みサイズ: $SZ bytes (正常なら約 53MB)"
  [ "$SZ" -lt 1000000 ] && echo "WARN: サイズが小さすぎます。DL失敗の可能性（HTMLが保存された等）"
fi

# ---- 4. 最終確認 -----------------------------------------------
banner "4. 確認"
python - <<'PY'
try:
    import diff_gaussian_rasterization
    print("[rasterizer] import OK ✅")
except Exception as e:
    print("[rasterizer] まだ NG ❌:", repr(e))
PY
echo "完了。問題が残っていれば上のログを共有してください。"
