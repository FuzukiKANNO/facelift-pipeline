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

# ---- パッチ: xformers 呼び出しを無効化 --------------------------
#   torch 2.11 に対応する xformers wheel が無いため。diffusers は
#   自動で PyTorch 標準の SDPA にフォールバックする（性能ほぼ同等）。
echo "[patch] FaceLift/inference.py の xformers を SDPA プロセッサへ差し替え..."
python - <<'PY'
p = "FaceLift/inference.py"
lines = open(p, encoding="utf-8").read().splitlines()
out, done = [], False
for ln in lines:
    if ("enable_xformers_memory_efficient_attention()" in ln) and ("set_attn_processor" not in ln) and not done:
        indent = ln[:len(ln) - len(ln.lstrip())]
        # xformers ではなく PyTorch SDPA ベースの効率アテンションを使う
        out.append(indent + "from diffusers.models.attention_processor import AttnProcessor2_0  # [patched-sdpa]")
        out.append(indent + "diffusion_pipeline.unet.set_attn_processor(AttnProcessor2_0())")
        done = True
    else:
        out.append(ln)
if done:
    open(p, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("  patched: xformers → SDPA(AttnProcessor2_0)")
else:
    print("  no change (already patched or line not found)")
PY

# ---- パッチ: GS-LRM の xformers 依存を SDPA に切替 ---------------
#   gslrm/model/utils_transformer.py は xformers を必須 import し、
#   use_flashatt_v2 時に flash attention を使う。xformers を任意依存化し、
#   未導入なら PyTorch SDPA パス（実装済み）を使うようにする。
echo "[patch] gslrm/model/utils_transformer.py の xformers 依存を SDPA へ..."
python - <<'PY'
import re
p = "FaceLift/gslrm/model/utils_transformer.py"
s = open(p, encoding="utf-8").read()
orig = s
# 1) import 失敗を致命的にしない（raise e → xops = None）
s = re.sub(
    r'except ImportError as e:\s*\n\s*print\([^)]*\)\s*\n\s*raise e',
    'except ImportError:\n    xops = None  # [patched] xformers を任意依存化',
    s, count=1)
# 2) xformers 未導入なら SDPA 分岐を使う
if 'xops is not None' not in s:
    s = s.replace('if self.use_flashatt_v2:',
                  'if self.use_flashatt_v2 and xops is not None:', 1)
if s != orig:
    open(p, "w", encoding="utf-8").write(s)
    print("  patched: xformers→SDPA")
else:
    print("  no change (already patched or pattern not found)")
PY

# ---- 1. FaceLift 推論（既存 .ply があればスキップ）--------------
EXISTING_PLY="$(find facelift_output/ -name '*.ply' 2>/dev/null | head -1)"
if [ -n "$EXISTING_PLY" ] && [ "${FORCE_INFER:-0}" != "1" ]; then
  echo "[1/3] 既存 .ply を再利用し推論をスキップ: $EXISTING_PLY"
  echo "      再推論する場合: FORCE_INFER=1 bash run_pipeline.sh"
else
  echo "[1/3] FaceLift 推論..."
  # メモリ断片化対策（OOM 回避のため）
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
fi

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
