#!/usr/bin/env python3
"""
recenter_gs_parts.py
パーツ別 3DGS (.ply) を「各パーツの重心が原点」に再センタリングし、
UnityGaussianSplatting が読める標準 INRIA 3DGS 形式で書き出す。
元の重心オフセットを manifest.json に保存（福笑いの正解配置に戻せる）。

標準レイアウト:
  x,y,z, nx,ny,nz(=0), f_dc_0..2, f_rest_0..K, opacity, scale_0..2, rot_0..3
（入力にある red/green/blue は Unity では不要なので落とす）

平行移動のみなので scale/rot/SH/opacity は不変（見た目は変わらない）。

使用例:
  python scripts/recenter_gs_parts.py \
    --parts_dir segmented_fukuwarai \
    --output_dir unity_parts
"""
import argparse
import glob
import json
import os
import numpy as np
from plyfile import PlyData, PlyElement


def collect_props(vertex):
    names = [p.name for p in vertex.properties]
    frest = sorted([n for n in names if n.startswith("f_rest_")],
                   key=lambda s: int(s.split("_")[-1]))
    return names, frest


def clean_mask(xyz):
    """
    誤割当の浮いたスプラット(floater)を除去するマスクを返す。
    統計的外れ値除去 → DBSCAN で最大クラスタのみ残す。
    open3d が無い場合は MAD ベースの簡易除去にフォールバック。
    """
    try:
        import open3d as o3d
    except Exception:
        med = np.median(xyz, 0)
        d = np.linalg.norm(xyz - med, axis=1)
        mad = np.median(np.abs(d - np.median(d))) + 1e-9
        return d < np.median(d) + 6.0 * mad

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    idx_all = np.arange(len(xyz))
    # 1) 統計的外れ値除去
    _, keep = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    keep = np.asarray(keep, dtype=int)
    sub = xyz[keep]
    # 2) DBSCAN で最大クラスタ
    p2 = o3d.geometry.PointCloud()
    p2.points = o3d.utility.Vector3dVector(sub.astype(np.float64))
    # eps は最近傍平均距離から推定
    dists = np.asarray(p2.compute_nearest_neighbor_distance())
    eps = float(np.median(dists)) * 3.0 if len(dists) else 0.02
    labels = np.asarray(p2.cluster_dbscan(eps=eps, min_points=10))
    mask = np.zeros(len(xyz), dtype=bool)
    if labels.max() >= 0:
        counts = np.bincount(labels[labels >= 0])
        biggest = counts.argmax()
        mask[keep[labels == biggest]] = True
    else:
        mask[keep] = True
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--parts", nargs="*", default=None)
    ap.add_argument("--pivot", choices=["median", "mean", "bbox"], default="median",
                    help="原点に合わせる基準（既定 median=外れ値に強い）")
    ap.add_argument("--no_clean", action="store_true",
                    help="floater(誤割当の浮いたスプラット)除去を無効化")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.parts:
        paths = [os.path.join(args.parts_dir, f"{p}.ply") for p in args.parts]
    else:
        paths = sorted(glob.glob(os.path.join(args.parts_dir, "*.ply")))

    manifest = {
        "note": "各パーツを重心原点に再センタリング。centroid_offset は元(共有)座標での位置。",
        "axis_note": "座標は 3DGS/FaceLift 系のまま（UnityGaussianSplatting が取込時に軸変換）。",
        "pivot": args.pivot,
        "parts": {},
    }

    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        if not os.path.exists(path):
            print(f"[skip] {name}: なし"); continue
        v = PlyData.read(path)["vertex"]
        names, frest = collect_props(v)
        xyz_full = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)

        # floater 除去
        if args.no_clean:
            keep = np.ones(len(xyz_full), dtype=bool)
        else:
            keep = clean_mask(xyz_full)
        n_removed = int((~keep).sum())
        xyz = xyz_full[keep]
        if len(xyz) == 0:
            print(f"[skip] {name}: クリーニングで全点除去"); continue

        if args.pivot == "median":
            offset = np.median(xyz, axis=0)
        elif args.pivot == "mean":
            offset = xyz.mean(0)
        else:  # bbox center
            offset = (xyz.min(0) + xyz.max(0)) / 2.0
        xyz_c = xyz - offset

        # 標準レイアウトの dtype
        fields = ["x", "y", "z", "nx", "ny", "nz",
                  "f_dc_0", "f_dc_1", "f_dc_2"] + frest + \
                 ["opacity", "scale_0", "scale_1", "scale_2",
                  "rot_0", "rot_1", "rot_2", "rot_3"]
        dtype = [(f, "f4") for f in fields]
        n = len(xyz_c)
        arr = np.zeros(n, dtype=dtype)
        arr["x"], arr["y"], arr["z"] = xyz_c[:, 0], xyz_c[:, 1], xyz_c[:, 2]
        # nx,ny,nz は 0 のまま
        for f in fields[6:]:
            if f in names:
                arr[f] = np.asarray(v[f]).astype(np.float32)[keep]
            # 入力に無ければ 0（通常は全て存在）

        el = PlyElement.describe(arr, "vertex")
        out = os.path.join(args.output_dir, f"{name}.ply")
        PlyData([el], text=False).write(out)

        bbox = (xyz.max(0) - xyz.min(0))
        manifest["parts"][name] = {
            "file": os.path.basename(out),
            "num_gaussians": int(n),
            "num_removed_floaters": n_removed,
            "centroid_offset": [round(float(x), 6) for x in offset],
            "bbox_size": [round(float(x), 6) for x in bbox],
        }
        print(f"[OK] {name}: {n:,} splats (floater除去 {n_removed})  "
              f"offset=({offset[0]:.3f},{offset[1]:.3f},{offset[2]:.3f}) "
              f"bbox=({bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f})")

    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n完了: {args.output_dir}（{len(manifest['parts'])} パーツ + manifest.json）")


if __name__ == "__main__":
    main()
