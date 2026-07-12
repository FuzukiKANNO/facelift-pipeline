#!/usr/bin/env python3
"""
diagnose_projection.py
FaceLift 出力 .ply の座標系を調べる。点群を各平面(XY/XZ/ZY/YZ)に
投影した画像を1枚に並べて保存する。どのパネルが「直立した正面顔」に
見えるかで、正しい投影軸を決める。

使い方:
  python scripts/diagnose_projection.py --ply facelift_output/.../gaussians.ply
"""
import argparse
import numpy as np
from plyfile import PlyData
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", default="segmented_output/projection_debug.png")
    args = ap.parse_args()

    v = PlyData.read(args.ply)["vertex"]
    x, y, z = np.array(v["x"]), np.array(v["y"]), np.array(v["z"])
    P = np.stack([x, y, z], axis=1).astype(np.float64)
    n = len(x)
    print(f"N = {n:,}")
    for i, ax in enumerate("xyz"):
        col = P[:, i]
        lo, hi = np.percentile(col, [1, 99])
        print(f"  {ax}: min={col.min():8.3f} max={col.max():8.3f} "
              f"p1={lo:8.3f} p99={hi:8.3f} range(1-99)={hi-lo:7.3f}")

    # 外れ値を除いた範囲でクリップ表示
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=min(n, 50000), replace=False)
    Ps = P[idx]

    # (i軸=横, j軸=縦) の4通り
    pairs = [(0, 1, "XY"), (0, 2, "XZ"), (2, 1, "ZY"), (1, 2, "YZ")]
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    for a, (i, j, name) in zip(axes, pairs):
        a.scatter(Ps[:, i], Ps[:, j], s=0.4, alpha=0.25, c="k")
        a.set_title(name, fontsize=16)
        a.set_xlabel("xyz"[i]); a.set_ylabel("xyz"[j])
        a.set_aspect("equal")
        a.invert_yaxis()  # 画像座標に合わせて縦反転
    plt.tight_layout()
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=90)
    print(f"\n保存: {args.out}")
    print("→ この画像を開き、『直立した正面の顔』に見えるパネル(XY/XZ/ZY/YZ)を確認してください。")


if __name__ == "__main__":
    main()
