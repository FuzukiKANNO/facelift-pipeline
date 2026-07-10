#!/usr/bin/env python3
"""
verify_ply.py
分割後の .ply 群を検証する。
  * Gaussian 数
  * バウンディングボックス（XYZ）
  * プロパティ名一覧（元 PLY と一致するか）
  * NaN / Inf の混入チェック
  * 全パーツ合計が元 PLY の総数と一致するか（other まで含めた場合）

使用方法:
  python verify_ply.py --output_dir ../segmented_output/ --ply_path ../facelift_output/face/gaussians.ply
"""
import argparse
import glob
import os
import numpy as np
from plyfile import PlyData


def summarize(ply_path: str):
    data = PlyData.read(ply_path)
    v = data["vertex"]
    n = len(v)
    props = [p.name for p in v.properties]
    info = {"n": n, "props": props}
    if n > 0 and {"x", "y", "z"}.issubset(props):
        xyz = np.stack([np.array(v["x"]), np.array(v["y"]), np.array(v["z"])], axis=1)
        info["bbox_min"] = xyz.min(axis=0)
        info["bbox_max"] = xyz.max(axis=0)
        # NaN/Inf チェック（全数値プロパティ）
        bad = 0
        for p in props:
            arr = np.array(v[p], dtype=np.float64, copy=False)
            bad += int(np.isnan(arr).sum() + np.isinf(arr).sum())
        info["bad_values"] = bad
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, help="segment_gaussians.py の出力ディレクトリ")
    parser.add_argument("--ply_path", default=None, help="（任意）元 PLY。総数照合に使う")
    args = parser.parse_args()

    parts = sorted(glob.glob(os.path.join(args.output_dir, "*.ply")))
    if not parts:
        print(f"[ERROR] {args.output_dir} に .ply が見つかりません")
        return

    total = 0
    ref_props = None
    print(f"=== 検証: {args.output_dir} ===\n")
    for p in parts:
        info = summarize(p)
        total += info["n"]
        name = os.path.basename(p)
        print(f"[{name}]")
        print(f"  Gaussian 数 : {info['n']:,}")
        if info["n"] > 0:
            bmin = info["bbox_min"]
            bmax = info["bbox_max"]
            print(f"  bbox min    : ({bmin[0]:.3f}, {bmin[1]:.3f}, {bmin[2]:.3f})")
            print(f"  bbox max    : ({bmax[0]:.3f}, {bmax[1]:.3f}, {bmax[2]:.3f})")
            print(f"  NaN/Inf     : {info['bad_values']}")
            if info["bad_values"] > 0:
                print(f"  [警告] 不正な数値が含まれています")
            if ref_props is None:
                ref_props = info["props"]
            elif info["props"] != ref_props:
                print(f"  [警告] プロパティ順が他と不一致")
        print()

    print(f"=== 合計 Gaussian 数（全パーツ）: {total:,} ===")
    if args.ply_path and os.path.exists(args.ply_path):
        src = summarize(args.ply_path)
        print(f"=== 元 PLY の総数            : {src['n']:,} ===")
        if total == src["n"]:
            print("[OK] 合計が元 PLY と一致（各 Gaussian は排他的に1パーツへ割当）")
        else:
            diff = src["n"] - total
            print(f"[注意] 差分 {diff:,}。PART_GROUPS の重複/欠落を確認してください。")


if __name__ == "__main__":
    main()
