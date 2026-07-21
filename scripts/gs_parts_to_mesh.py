#!/usr/bin/env python3
"""
gs_parts_to_mesh.py
パーツ別 3DGS (.ply) を Unity で扱えるメッシュ (.glb / .obj) に変換する。

処理:
  1. 各パーツ .ply の Gaussian 中心を点群化し、SH 基底(f_dc)から頂点色を計算
  2. 法線推定 → Poisson 表面再構成 → 低密度面除去 → 最大連結成分抽出 → Taubin 平滑化
  3. 最近傍 Gaussian の色を頂点色として転写
  4. FaceLift 前面カメラ基準の「直立右手系 Y-up」座標へ変換（Unity で上下正しく入る）
  5. 各パーツを自身の重心が原点になるよう再センタリング（掴んで動かしやすい）
  6. 元の相対位置（重心オフセット）を manifest.json に保存（正しい配置に戻せる／福笑い）
  7. .glb（頂点色つき）と .obj（形状のみ）を書き出し

使用例:
  python scripts/gs_parts_to_mesh.py \
    --parts_dir segmented_output \
    --camera_json FaceLift/utils_folder/opencv_cameras.json \
    --camera_index 2 \
    --output_dir mesh_output

Unity 取り込み:
  .glb は glTFast（Unity Registry から無料）でインポートすると頂点色つきで入る。
  .obj は標準インポートで形状のみ（色なし）。manifest.json の centroid_offset に
  GameObject を置くと元の顔配置に復元できる（福笑いの「正解位置」）。
"""
import argparse
import json
import os
import glob
import numpy as np
import open3d as o3d
from plyfile import PlyData

SH_C0 = 0.28209479177387814  # 球面調和 0次の係数


def load_gaussian_ply(path):
    pd = PlyData.read(path)["vertex"]
    pos = np.stack([np.array(pd["x"]), np.array(pd["y"]), np.array(pd["z"])], 1).astype(np.float64)
    dc = np.stack([np.array(pd["f_dc_0"]), np.array(pd["f_dc_1"]), np.array(pd["f_dc_2"])], 1).astype(np.float64)
    rgb = np.clip(SH_C0 * dc + 0.5, 0.0, 1.0)
    return pos, rgb


def front_camera(camera_json, camera_index):
    c = json.load(open(camera_json))["frames"][camera_index]
    w2c = np.array(c["w2c"], np.float64)
    c2w = np.linalg.inv(w2c)
    cam_loc = c2w[:3, 3]
    return w2c, cam_loc


def to_upright_frame(pos_world, w2c):
    """
    world -> 前面カメラ基準の直立右手系 Y-up へ。
      cam = w2c @ [X;1]  (x=右, y=下, z=前方/奥)
      upright: X=x, Y=-y(上), Z=-z(手前が +Z=正面)
    右手系のまま（行列式 +1）なので鏡像反転しない。glTF/OBJ とも上下正しく入る。
    """
    n = pos_world.shape[0]
    ph = np.concatenate([pos_world, np.ones((n, 1))], 1)
    cam = (w2c @ ph.T).T[:, :3]
    up = np.stack([cam[:, 0], -cam[:, 1], -cam[:, 2]], 1)
    return up


def reconstruct_mesh(pos, rgb, cam_loc_upright, depth, density_quantile, smooth_iters):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    pcd.orient_normals_towards_camera_location(cam_loc_upright)

    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    dens = np.asarray(dens)
    if len(dens) > 0:
        mesh.remove_vertices_by_mask(dens < np.quantile(dens, density_quantile))
        mesh.remove_unreferenced_vertices()
    # 浮いた小クラスタ除去 → 最大連結成分のみ
    tclu, ntri, _ = mesh.cluster_connected_triangles()
    tclu = np.asarray(tclu); ntri = np.asarray(ntri)
    if len(ntri) > 0:
        mesh.remove_triangles_by_mask(tclu != int(ntri.argmax()))
        mesh.remove_unreferenced_vertices()
    if smooth_iters > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iters)
    mesh.compute_vertex_normals()

    # 頂点色: 最近傍 Gaussian
    kdt = o3d.geometry.KDTreeFlann(pcd)
    mv = np.asarray(mesh.vertices)
    vcol = np.zeros((len(mv), 3))
    for i, p in enumerate(mv):
        _, idx, _ = kdt.search_knn_vector_3d(p, 1)
        vcol[i] = rgb[idx[0]]
    mesh.vertex_colors = o3d.utility.Vector3dVector(vcol)
    return mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts_dir", required=True, help="パーツ .ply があるディレクトリ")
    ap.add_argument("--camera_json", required=True, help="opencv_cameras.json")
    ap.add_argument("--camera_index", type=int, default=2, help="前面カメラ frame（既定 2）")
    ap.add_argument("--output_dir", required=True, help="メッシュ出力先")
    ap.add_argument("--parts", nargs="*", default=None,
                    help="対象パーツ名（未指定なら parts_dir の全 .ply）")
    ap.add_argument("--depth", type=int, default=9, help="Poisson 深さ")
    ap.add_argument("--density_quantile", type=float, default=0.2, help="低密度面の除去分位")
    ap.add_argument("--smooth_iters", type=int, default=15, help="Taubin 平滑化回数")
    ap.add_argument("--min_gaussians", type=int, default=80, help="この数未満のパーツはスキップ")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    w2c, cam_loc_world = front_camera(args.camera_json, args.camera_index)
    # カメラ位置も直立フレームへ（法線向き付け用）
    cam_loc_upright = to_upright_frame(cam_loc_world[None, :], w2c)[0]

    if args.parts:
        ply_paths = [os.path.join(args.parts_dir, f"{p}.ply") for p in args.parts]
    else:
        ply_paths = sorted(glob.glob(os.path.join(args.parts_dir, "*.ply")))

    manifest = {
        "coordinate_note": "front-camera upright, right-handed Y-up (+Z=正面/手前). Unity は glTFast で取込むと上下正しい。",
        "unit_note": "FaceLift ワールド単位。頭部が概ね半径〜1。Unity で実寸に合わせてスケール。",
        "camera_index": args.camera_index,
        "parts": {},
    }

    for p in ply_paths:
        name = os.path.splitext(os.path.basename(p))[0]
        if not os.path.exists(p):
            print(f"[skip] {name}: ファイルなし"); continue
        pos_w, rgb = load_gaussian_ply(p)
        if len(pos_w) < args.min_gaussians:
            print(f"[skip] {name}: Gaussian {len(pos_w)} < {args.min_gaussians}"); continue
        pos = to_upright_frame(pos_w, w2c)
        mesh = reconstruct_mesh(pos, rgb, cam_loc_upright,
                                args.depth, args.density_quantile, args.smooth_iters)
        V = np.asarray(mesh.vertices)
        if len(V) == 0:
            print(f"[skip] {name}: メッシュ生成できず"); continue

        # 再センタリング（重心を原点へ）。offset は共有フレームでの元位置。
        centroid = V.mean(0)
        mesh.vertices = o3d.utility.Vector3dVector(V - centroid)
        bbox = (V.max(0) - V.min(0))

        glb = os.path.join(args.output_dir, f"{name}.glb")
        obj = os.path.join(args.output_dir, f"{name}.obj")
        o3d.io.write_triangle_mesh(glb, mesh)
        o3d.io.write_triangle_mesh(obj, mesh)

        manifest["parts"][name] = {
            "file_glb": os.path.basename(glb),
            "file_obj": os.path.basename(obj),
            "centroid_offset": [round(float(x), 6) for x in centroid],
            "bbox_size": [round(float(x), 6) for x in bbox],
            "num_gaussians": int(len(pos_w)),
            "num_vertices": int(len(V)),
            "num_triangles": int(len(np.asarray(mesh.triangles))),
        }
        print(f"[OK] {name}: V={len(V):,} F={len(np.asarray(mesh.triangles)):,} "
              f"offset=({centroid[0]:.3f},{centroid[1]:.3f},{centroid[2]:.3f})")

    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n完了: {args.output_dir} に {len(manifest['parts'])} パーツ + manifest.json")


if __name__ == "__main__":
    main()
