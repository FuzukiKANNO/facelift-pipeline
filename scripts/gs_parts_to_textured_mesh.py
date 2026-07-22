#!/usr/bin/env python3
"""
gs_parts_to_textured_mesh.py
パーツ別 3DGS を「メッシュ＋実写テクスチャ」に変換する（GSより拡大に強くくっきり）。

処理:
  1. パーツ .ply の Gaussian 中心 → floater除去 → Poisson 表面再構成 → 最大連結成分 → 平滑化
  2. FaceLift 前面カメラ(opencv_cameras.json frame)で各頂点を UV 化
  3. 前面カメラに整合した実写 (facelift_output/<name>/input.png) をテクスチャとして貼る
  4. 各パーツを重心原点に再センタリング（掴みやすいピボット）、元位置は manifest に保存
  5. obj + mtl + png（Unity標準インポート）で書き出し

使用例:
  python scripts/gs_parts_to_textured_mesh.py \
    --parts_dir segmented_fukuwarai \
    --camera_json FaceLift/utils_folder/opencv_cameras.json --camera_index 2 \
    --texture facelift_output/face/input.png \
    --output_dir textured_parts
"""
import argparse, json, os, glob
import numpy as np, cv2, open3d as o3d
from plyfile import PlyData


def clean_mask(xyz):
    p = o3d.geometry.PointCloud(); p.points = o3d.utility.Vector3dVector(xyz)
    _, keep = p.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    keep = np.asarray(keep, int)
    sub = xyz[keep]
    p2 = o3d.geometry.PointCloud(); p2.points = o3d.utility.Vector3dVector(sub)
    d = np.asarray(p2.compute_nearest_neighbor_distance())
    eps = float(np.median(d)) * 3 if len(d) else 0.02
    lab = np.asarray(p2.cluster_dbscan(eps=eps, min_points=10))
    m = np.zeros(len(xyz), bool)
    if lab.max() >= 0:
        m[keep[lab == np.bincount(lab[lab >= 0]).argmax()]] = True  # 最大クラスタのみ（単一塊を想定）
    else:
        m[keep] = True
    return m


def build(name, ply, cam, tex_bgr, depth, dq, smooth, fill_holes=0.0):
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    W, H = int(cam.get("w", 512)), int(cam.get("h", 512))
    w2c = np.array(cam["w2c"], float)

    pd = PlyData.read(ply)["vertex"]
    xyz = np.stack([pd["x"], pd["y"], pd["z"]], 1).astype(np.float64)
    m = clean_mask(xyz); xyz = xyz[m]
    if len(xyz) < 50:
        print(f"[skip] {name}: too few points"); return None

    pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(xyz)
    camloc = np.linalg.inv(w2c)[:3, 3]
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    pcd.orient_normals_towards_camera_location(camloc)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, dq)); mesh.remove_unreferenced_vertices()
    tclu, ntri, _ = mesh.cluster_connected_triangles(); tclu = np.asarray(tclu); ntri = np.asarray(ntri)
    if len(ntri):
        mesh.remove_triangles_by_mask(tclu != int(ntri.argmax())); mesh.remove_unreferenced_vertices()  # 最大成分のみ
    if smooth > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth)
    # 小さな穴（鼻孔など）を塞ぐ。外周(大きな境界)は残すよう hole_size を制限。
    if fill_holes > 0:
        try:
            diag0 = float(np.linalg.norm(np.asarray(mesh.get_max_bound()) - np.asarray(mesh.get_min_bound())))
            tm = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            tm = tm.fill_holes(hole_size=diag0 * fill_holes)
            mesh = tm.to_legacy()
        except Exception as e:
            print(f"  [warn] fill_holes skipped: {e}")
    mesh.compute_vertex_normals()

    V = np.asarray(mesh.vertices); T = np.asarray(mesh.triangles)
    # 前面カメラで UV（画像は上が v=0）。Unity は uv.y=0 が下なので 1-v。
    ph = np.concatenate([V, np.ones((len(V), 1))], 1); c = (w2c @ ph.T).T
    u = fx * c[:, 0] / c[:, 2] + cx; v = fy * c[:, 1] / c[:, 2] + cy
    uv = np.stack([np.clip(u / W, 0, 1), 1.0 - np.clip(v / H, 0, 1)], 1)

    centroid = np.median(V, axis=0)
    Vc = V - centroid
    bbox = V.max(0) - V.min(0)
    mesh.vertices = o3d.utility.Vector3dVector(Vc)
    mesh.triangle_uvs = o3d.utility.Vector2dVector(uv[T].reshape(-1, 2))
    mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros(len(T), np.int32))
    mesh.textures = [o3d.geometry.Image(np.ascontiguousarray(tex_bgr[:, :, ::-1]))]  # RGB

    # 境界フェード用アルファ: 外周(境界)に近い頂点ほど透明に。
    # 境界エッジ = 1三角形にしか属さないエッジ。
    from collections import defaultdict
    ecount = defaultdict(int)
    for tri in T:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            ecount[(min(a, b), max(a, b))] += 1
    bnd = set()
    for (a, b), cnt in ecount.items():
        if cnt == 1:
            bnd.add(a); bnd.add(b)
    diag = float(np.linalg.norm(bbox))
    feather = max(diag * 0.07, 1e-6)   # 縁だけ薄く（透明すぎ防止）
    if bnd:
        bpts = Vc[sorted(bnd)]
        bpcd = o3d.geometry.PointCloud(); bpcd.points = o3d.utility.Vector3dVector(bpts)
        bkdt = o3d.geometry.KDTreeFlann(bpcd)
        d = np.empty(len(Vc))
        for i, p in enumerate(Vc):
            _, _, dist2 = bkdt.search_knn_vector_3d(p, 1)
            d[i] = np.sqrt(dist2[0]) if len(dist2) else feather
        alpha = np.clip(d / feather, 0.0, 1.0)
    else:
        alpha = np.ones(len(Vc))

    # --- Unity 直接生成用データ（OBJインポータを介さず座標を完全制御）---
    # world -> 前面カメラ基準の直立 (x右, y上, z手前)。さらに右手→左手系へ z 反転。
    R = w2c[:3, :3]
    Mrot = np.diag([1.0, -1.0, -1.0]) @ R          # world -> upright(RH, +Z手前)
    flip = np.array([1.0, 1.0, -1.0])              # RH -> Unity LH
    Uverts = (Mrot @ Vc.T).T * flip                # (N,3) 直立・左手系・重心原点
    Uoff = (Mrot @ centroid) * flip                # 元位置（同フレーム）
    unity = {
        "verts": [round(float(x), 6) for x in Uverts.reshape(-1)],
        "uvs":   [round(float(x), 6) for x in uv.reshape(-1)],
        "tris":  [int(x) for x in T.reshape(-1)],
        "alpha": [round(float(x), 4) for x in alpha],
        "offset": [round(float(x), 6) for x in Uoff],
        "bbox":  [round(float(x), 6) for x in bbox],
    }
    return mesh, centroid, bbox, len(V), len(T), unity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts_dir", required=True)
    ap.add_argument("--camera_json", required=True)
    ap.add_argument("--camera_index", type=int, default=2)
    ap.add_argument("--texture", required=True, help="前面カメラに整合した実写 (input.png)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--parts", nargs="*", default=None)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--density_quantile", type=float, default=0.1)
    ap.add_argument("--smooth_iters", type=int, default=15)
    ap.add_argument("--fill_holes", type=float, default=0.0,
                    help="小さな穴を塞ぐ最大境界長（bbox対角比）。0で無効（外周まで塞ぐと壊れるため既定OFF）")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cam = json.load(open(args.camera_json))["frames"][args.camera_index]
    tex = cv2.imread(args.texture)
    if tex is None:
        raise SystemExit(f"texture not found: {args.texture}")
    W, H = int(cam.get("w", 512)), int(cam.get("h", 512))
    tex = cv2.resize(tex, (W, H))

    paths = ([os.path.join(args.parts_dir, f"{p}.ply") for p in args.parts]
             if args.parts else sorted(glob.glob(os.path.join(args.parts_dir, "*.ply"))))
    manifest = {"note": "textured mesh (obj+png). front-camera UV. centroid_offset=元位置。",
                "camera_index": args.camera_index, "parts": {}}
    for ply in paths:
        name = os.path.splitext(os.path.basename(ply))[0]
        if not os.path.exists(ply):
            print(f"[skip] {name}: なし"); continue
        r = build(name, ply, cam, tex, args.depth, args.density_quantile, args.smooth_iters, args.fill_holes)
        if r is None:
            continue
        mesh, centroid, bbox, nv, nt, unity = r
        pdir = os.path.join(args.output_dir, name); os.makedirs(pdir, exist_ok=True)
        out = os.path.join(pdir, f"{name}.obj")
        o3d.io.write_triangle_mesh(out, mesh, write_triangle_uvs=True)
        # Unity 直接生成用: メッシュデータ JSON + テクスチャ png
        json.dump(unity, open(os.path.join(pdir, f"{name}.meshdata.json"), "w"), separators=(",", ":"))
        cv2.imwrite(os.path.join(pdir, f"{name}_tex.png"), tex)
        manifest["parts"][name] = {
            "dir": name, "obj": f"{name}.obj",
            "meshdata": f"{name}.meshdata.json", "texture": f"{name}_tex.png",
            "centroid_offset": [round(float(x), 6) for x in centroid],
            "bbox_size": [round(float(x), 6) for x in bbox],
            "num_vertices": int(nv), "num_triangles": int(nt),
        }
        print(f"[OK] {name}: V={nv:,} F={nt:,} -> {out}")

    json.dump(manifest, open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n完了: {args.output_dir}（{len(manifest['parts'])} パーツ）")


if __name__ == "__main__":
    main()
