#!/usr/bin/env python3
"""
segment_gaussians.py
FaceLift で生成した 3DGS (.ply) を BiSeNet 顔パース結果で
パーツ別 .ply に分割するスクリプト（バグ修正版）

使用方法:
  python segment_gaussians.py \
    --ply_path ../facelift_output/face/gaussians.ply \
    --face_image ../input/face.jpg \
    --face_parse_root ../face-parsing.PyTorch \
    --output_dir ../segmented_output/

主な修正点（元指示書からの変更）:
  * save_ply_subset() の壊れた vertex_data ブロックを削除。
    PLY の全プロパティ（f_dc/f_rest/opacity/scale/rot 等）を
    元の順序・dtype のまま保持して書き出す。
  * torch.load を weights_only=False で明示（PyTorch>=2.6 対策）。
  * 投影軸の反転を --flip_x / --flip_y フラグで切替可能に。
  * ラベルマップの解像度は正方形 512 のまま扱い、投影も同じ座標系で行う
    （元解像度リサイズ由来のアスペクト比ずれを避ける）。
"""
import argparse
import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image
from plyfile import PlyData, PlyElement

# ============================================================
# BiSeNet 19クラス定義（CelebAMask-HQ）
# ============================================================
BISENET_CLASSES = {
    0:  "background",
    1:  "skin",
    2:  "l_brow",
    3:  "r_brow",
    4:  "l_eye",
    5:  "r_eye",
    6:  "eye_g",      # メガネ
    7:  "l_ear",
    8:  "r_ear",
    9:  "ear_r",      # イヤリング
    10: "nose",
    11: "mouth",
    12: "u_lip",
    13: "l_lip",
    14: "neck",
    15: "neck_l",     # ネックレス
    16: "cloth",
    17: "hair",
    18: "hat",
}

# パーツのグループ定義（出力する .ply ファイルに対応）
PART_GROUPS = {
    "eye_left":   [4],                # 左目
    "eye_right":  [5],                # 右目
    "eyebrow":    [2, 3],             # 眉
    "nose":       [10],               # 鼻
    "mouth":      [11, 12, 13],       # 口・唇
    "skin_cheek": [1],                # 肌・頬
    "ear":        [7, 8, 9],          # 耳
    "hair":       [17, 18],           # 髪・帽子
    "other":      [0, 6, 14, 15, 16], # 背景・その他
}


# ============================================================
# BiSeNet 推論
# ============================================================
def run_bisenet(image_path: str, parse_root: str, device: str = "cuda") -> np.ndarray:
    """
    BiSeNet で顔パース → ラベルマップを返す
    戻り値: shape (512, 512) の uint8 配列（0〜18 のクラスID）
    投影と同じ 512x512 座標系で扱うため、元解像度へは戻さない。
    """
    sys.path.insert(0, parse_root)
    from model import BiSeNet  # face-parsing.PyTorch の model.py

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.to(device)

    model_path = os.path.join(parse_root, "res/cp/79999_iter.pth")
    net.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    net.eval()

    # 前処理
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((512, 512), Image.BILINEAR)
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np - mean) / std
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        out = net(img_tensor)[0]
    parsing = out.squeeze(0).cpu().numpy().argmax(0)  # (512, 512)
    return parsing.astype(np.uint8)


# ============================================================
# PLY 読み込み
# ============================================================
def load_ply(ply_path: str):
    """
    .ply を読み込み、頂点データを dict（プロパティ名 -> ndarray）で返す。
    dict はファイル内のプロパティ順を保持する（Python 3.7+）。
    """
    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"]
    props = {prop.name: np.array(vertex[prop.name]) for prop in vertex.properties}
    return props, plydata


# ============================================================
# Gaussian の 2D 投影
# ============================================================
def project_gaussians_to_image(
    positions_xyz: np.ndarray,
    image_w: int,
    image_h: int,
    h_axis: int = 0,
    v_axis: int = 2,
    flip_h: bool = False,
    flip_v: bool = True,
) -> np.ndarray:
    """
    Gaussian を指定 2 軸で画像座標に正射影する（簡易正投影）。
    percentile で外れ値を除いた bbox に正規化する。

    FaceLift 出力は「顔が XZ 平面・Y が奥行き」なので既定は
    h_axis=0(X, 横) / v_axis=2(Z, 縦)。flip_v=True で頭を上に向ける。

    h_axis, v_axis: 0=x, 1=y, 2=z
    flip_h: 横方向を反転（左右ミラー）
    flip_v: 縦方向を反転（画像座標系 py=0 が上）

    戻り値: shape (N, 2) の float32 配列（画像ピクセル座標）
    """
    hs = positions_xyz[:, h_axis]
    vs = positions_xyz[:, v_axis]
    h_min, h_max = np.percentile(hs, 1), np.percentile(hs, 99)
    v_min, v_max = np.percentile(vs, 1), np.percentile(vs, 99)

    nh = np.clip((hs - h_min) / (h_max - h_min + 1e-8), 0, 1)
    nv = np.clip((vs - v_min) / (v_max - v_min + 1e-8), 0, 1)

    if flip_h:
        nh = 1.0 - nh
    if flip_v:
        nv = 1.0 - nv

    px = nh * (image_w - 1)
    py = nv * (image_h - 1)
    return np.stack([px, py], axis=1).astype(np.float32)


# ============================================================
# ラベル割り当て
# ============================================================
def assign_labels(pixel_coords: np.ndarray, label_map: np.ndarray) -> np.ndarray:
    """
    各 Gaussian のピクセル座標からラベルを取得する
    戻り値: shape (N,) の int 配列（0〜18）
    """
    px = np.clip(pixel_coords[:, 0].astype(int), 0, label_map.shape[1] - 1)
    py = np.clip(pixel_coords[:, 1].astype(int), 0, label_map.shape[0] - 1)
    return label_map[py, px]


# ============================================================
# PLY 書き出し
# ============================================================
def save_ply_subset(props: dict, mask: np.ndarray, output_path: str):
    """
    mask が True の Gaussian だけを抽出して .ply に保存する。
    全プロパティを元の順序・dtype（float64 は float32 に縮約）で保持する。
    """
    n = int(mask.sum())
    if n == 0:
        print(f"  [SKIP] {os.path.basename(output_path)}: Gaussian が 0 個")
        return

    dtypes = []
    arrays = {}
    for key, val in props.items():
        arr = val[mask]
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        arrays[key] = arr
        dtypes.append((key, arr.dtype))

    vertex_array = np.zeros(n, dtype=dtypes)
    for key in arrays:
        vertex_array[key] = arrays[key]

    el = PlyElement.describe(vertex_array, "vertex")
    # 3DGS ビューア互換のためバイナリで書き出す
    PlyData([el], text=False).write(output_path)
    print(f"  [OK] {os.path.basename(output_path)}: {n:,} Gaussians")


# ============================================================
# メイン処理
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply_path", required=True, help="FaceLift 出力の .ply パス")
    parser.add_argument("--face_image", required=True, help="入力顔画像パス")
    parser.add_argument("--face_parse_root", required=True, help="face-parsing.PyTorch のルートパス")
    parser.add_argument("--output_dir", required=True, help="出力ディレクトリ")
    parser.add_argument("--device", default="cuda", help="cuda / cpu")
    parser.add_argument("--h_axis", type=int, default=0, help="横に使う軸 0=x,1=y,2=z（既定 x）")
    parser.add_argument("--v_axis", type=int, default=2, help="縦に使う軸 0=x,1=y,2=z（既定 z）")
    parser.add_argument("--flip_h", action="store_true", help="横方向を反転（左右ミラー）")
    parser.add_argument("--no_flip_v", action="store_true", help="縦方向の反転をやめる")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. BiSeNet で顔パース
    print("[1/4] BiSeNet 顔パース実行中...")
    label_map = run_bisenet(args.face_image, args.face_parse_root, args.device)
    image_h, image_w = label_map.shape
    print(f"  ラベルマップ解像度: {image_w} x {image_h}")

    # パースマスク可視化（デバッグ用）
    debug_path = os.path.join(args.output_dir, "debug_label_map.png")
    label_vis = (label_map.astype(np.float32) / 18.0 * 255).astype(np.uint8)
    label_color = cv2.applyColorMap(label_vis, cv2.COLORMAP_JET)
    cv2.imwrite(debug_path, label_color)
    print(f"  ラベルマップ保存: {debug_path}")

    # 2. PLY 読み込み
    print("[2/4] PLY 読み込み中...")
    props, _ = load_ply(args.ply_path)
    if "x" not in props:
        print(f"  [ERROR] PLY に 'x' プロパティがありません。keys={list(props.keys())}")
        sys.exit(1)
    n_total = len(props["x"])
    print(f"  総 Gaussian 数: {n_total:,}")
    print(f"  PLY プロパティ: {list(props.keys())}")
    positions = np.stack([props["x"], props["y"], props["z"]], axis=1)

    # 3. Gaussian を 2D 投影してラベル取得
    print("[3/4] Gaussian を 2D 投影してラベル割り当て中...")
    axis_name = "xyz"
    print(f"  投影軸: 横={axis_name[args.h_axis]} 縦={axis_name[args.v_axis]} "
          f"flip_h={args.flip_h} flip_v={not args.no_flip_v}")
    pixel_coords = project_gaussians_to_image(
        positions, image_w, image_h,
        h_axis=args.h_axis,
        v_axis=args.v_axis,
        flip_h=args.flip_h,
        flip_v=not args.no_flip_v,
    )
    labels = assign_labels(pixel_coords, label_map)

    # ラベル分布表示
    print("  ラベル分布（上位10クラス）:")
    unique, counts = np.unique(labels, return_counts=True)
    for cls_id, cnt in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
        print(f"    {cls_id:2d} {BISENET_CLASSES[cls_id]:12s}: {cnt:8,} ({cnt/n_total*100:.1f}%)")

    # 4. パーツ別 .ply 書き出し
    print("[4/4] パーツ別 .ply 書き出し中...")
    for part_name, class_ids in PART_GROUPS.items():
        mask = np.isin(labels, class_ids)
        output_path = os.path.join(args.output_dir, f"{part_name}.ply")
        save_ply_subset(props, mask, output_path)

    # 未割り当て確認（PART_GROUPS は全19クラスを網羅しているため通常 0）
    all_assigned = np.zeros(n_total, dtype=bool)
    for class_ids in PART_GROUPS.values():
        all_assigned |= np.isin(labels, class_ids)
    unassigned = int((~all_assigned).sum())
    if unassigned > 0:
        print(f"  [警告] 未割り当て Gaussian: {unassigned}")

    print("\n[完了] 出力先:", args.output_dir)
    print("Unity の UnityGaussianSplatting で各 .ply を個別にインポートしてください。")


if __name__ == "__main__":
    main()
