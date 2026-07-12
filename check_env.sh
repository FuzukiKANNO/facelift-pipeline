#!/usr/bin/env bash
# ============================================================
# check_env.sh  —  セットアップ結果の健全性チェック
#   cd ~/facelift-pipeline && git pull && bash check_env.sh
# ============================================================
set -uo pipefail
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE"
ENV_NAME="${FACELIFT_ENV:-facelift}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME" || { echo "NG: conda 環境 '$ENV_NAME' を activate できません"; exit 1; }

echo "=== conda env: $ENV_NAME / $(which python) ==="
python - <<'PY'
ok = True
try:
    import torch
    print("[torch]        version :", torch.__version__)
    print("[torch]        cuda ok :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[torch]        device  :", torch.cuda.get_device_name(0))
        cap = torch.cuda.get_device_capability(0)
        print("[torch]        capability:", cap, "(RTX 5090 は (12,0))")
        # 実際に GPU 計算が通るか（Blackwell カーネルの有無を確認）
        x = torch.randn(1024, 1024, device="cuda")
        y = (x @ x).sum().item()
        print("[torch]        matmul on cuda: OK")
    else:
        ok = False
        print("[torch]        cuda 使用不可 → NG")
except Exception as e:
    ok = False
    print("[torch]        ERROR:", repr(e))

try:
    import diff_gaussian_rasterization
    print("[rasterizer]   import  : OK")
except Exception as e:
    ok = False
    print("[rasterizer]   ERROR   :", repr(e))

for mod in ("transformers", "diffusers", "plyfile", "cv2"):
    try:
        m = __import__(mod)
        print(f"[{mod:12s}] import  : OK", getattr(m, "__version__", ""))
    except Exception as e:
        ok = False
        print(f"[{mod:12s}] ERROR   :", repr(e))

try:
    import xformers
    print("[xformers]     import  : OK", xformers.__version__)
except Exception:
    print("[xformers]     未導入（FaceLift が要求したら pip install -U xformers）")

print("\n=== 総合:", "OK ✅" if ok else "問題あり ❌", "===")
PY

echo ""
echo "=== ファイル確認 ==="
[ -d FaceLift ] && echo "FaceLift/ : あり" || echo "FaceLift/ : なし ❌"
[ -d face-parsing.PyTorch ] && echo "face-parsing.PyTorch/ : あり" || echo "face-parsing.PyTorch/ : なし ❌"
W=face-parsing.PyTorch/res/cp/79999_iter.pth
if [ -f "$W" ]; then echo "BiSeNet 重み : $(du -h "$W" | cut -f1)"; else echo "BiSeNet 重み : なし ❌"; fi
[ -f input/face.jpg ] && echo "input/face.jpg : あり" || echo "input/face.jpg : なし（推論前に配置してください）"
