# FaceLift → 顔パーツ別 3D Gaussian 分割パイプライン

1枚の顔画像から FaceLift で 3D Gaussian Splatting (.ply) を生成し、BiSeNet の
顔パース結果を使ってパーツ別 .ply に分割する一式です。

> **重要**: FaceLift は CUDA (NVIDIA GPU) 必須です。この一式は Windows PC 側で
> 準備し、**Linux + NVIDIA GPU マシンに転送して実行**する前提で作られています。

---

## 転送先マシンの前提

- Linux（Ubuntu 22.04 推奨）
- NVIDIA CUDA GPU / VRAM 16GB 以上推奨（A100 / RTX 3090 等）
- conda インストール済み
- Python 3.10（FaceLift の setup_env.sh が構築）

---

## ディレクトリ構成

```
facelift-pipeline/
├── README.md              # このファイル
├── setup.sh               # リポジトリ clone・conda 環境・重み DL（Linux）
├── run_pipeline.sh        # FaceLift 推論 → 分割 → 検証（Linux）
├── .gitignore
├── input/
│   └── face.jpg           # 入力顔画像（正面・クロップ済み）※各自配置
└── scripts/
    ├── segment_gaussians.py   # Gaussian ラベル付け・分割
    └── verify_ply.py          # 出力検証
```

`setup.sh` 実行後、以下が自動で追加されます:
`FaceLift/`, `face-parsing.PyTorch/`, `facelift_output/`, `segmented_output/`

---

## 手順（Linux GPU マシン上）

### 0. 転送
このフォルダごと GPU マシンにコピー（git push→pull、scp、rsync 等）。

### 1. 入力画像を配置
```bash
cp /path/to/your/face.jpg input/face.jpg
```

### 2. セットアップ
```bash
bash setup.sh
```
- FaceLift を clone し `setup_env.sh` で conda 環境を作成
- face-parsing.PyTorch を clone
- `plyfile opencv-python pillow gdown` を追加インストール
- BiSeNet 重み `79999_iter.pth` を `gdown` で取得

> conda 環境名が `facelift` でない場合は `conda env list` で確認し、
> `FACELIFT_ENV=<環境名> bash setup.sh` のように指定してください。
> （run_pipeline.sh も同じ環境変数を見ます）

### 3. 実行
```bash
bash run_pipeline.sh
```
FaceLift 推論 → `facelift_output/` 内の .ply を自動検出 → 分割 → 検証まで通します。

### 4. 結果確認
- `segmented_output/*.ply` … パーツ別 Gaussian
- `segmented_output/debug_label_map.png` … BiSeNet 顔パース結果（目視確認用）
- verify_ply.py が各 .ply の Gaussian 数・bbox・NaN 有無・合計照合を表示

---

## 投影がずれる場合

`debug_label_map.png` の顔パース自体は正しいのに Gaussian への割当がずれる場合、
投影軸を反転して再実行します（run_pipeline.sh の segment 呼び出しに付与）:

```bash
# 上下反転を無効化
conda run -n facelift python scripts/segment_gaussians.py ... --no_flip_y
# 左右ミラー
conda run -n facelift python scripts/segment_gaussians.py ... --flip_x
```

全 Gaussian が `other` に集中する場合は、FaceLift 出力が Z 正面軸でない可能性。
`segment_gaussians.py` 実行時に表示される XYZ bbox とラベル分布を確認してください。

---

## 元指示書からの主な修正

- **`save_ply_subset()` のバグ修正**: 元版にあった壊れた `vertex_data` ブロック
  （未使用でエラー要因）を削除。全プロパティ（`f_dc_*`, `f_rest_*`, `opacity`,
  `scale_*`, `rot_*` 等）を元順序・dtype のままバイナリ .ply として保存。
- **`verify_ply.py` を新規作成**（元指示書は参照のみで中身なし）。
- **`torch.load(..., weights_only=False)`** を明示（PyTorch ≥ 2.6 対策）。
- **ラベルマップを 512×512 のまま扱う**ことで、元解像度リサイズ由来のアスペクト
  比ずれを排除。投影も同じ座標系に統一。
- **投影軸反転を `--flip_x` / `--no_flip_y` フラグ化**。

---

## Unity での利用

1. [UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting) を導入
2. 各 `.ply` を `Assets/GaussianSplats/` に配置
3. パーツごとに GaussianSplatRenderer を作成し、対応する .ply をアサイン
4. パーツ単位で ON/OFF・移動・スケールできることを確認

---

## トラブルシューティング

| 症状 | 対処 |
|------|------|
| `model.py` が見つからない | `--face_parse_root` が `face-parsing.PyTorch/` を指すか確認 |
| PLY のプロパティ名が想定と違う | segment 実行時のログ「PLY プロパティ」を確認 |
| CUDA メモリ不足 | segment 側は `--device cpu` 可（FaceLift 本体は GPU 必須） |
| 全 Gaussian が other | 投影軸を確認（上記「投影がずれる場合」） |
| gdown ダウンロード失敗 | face-parsing.PyTorch の README リンクから手動 DL し `res/cp/79999_iter.pth` に配置 |
