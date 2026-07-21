# Unity で顔パーツ（Gaussian Splat）を扱う手順

3D福笑い／HMD重畳の準備。各顔パーツを **Gaussian Splat のまま** Unity に取り込み、
掴んで動かせるオブジェクトにする。想定環境は **PC 接続 HMD（このPCの GPU で描画）**。

> メッシュ化も試したが、鼻孔・口内・眉など暗く小さいパーツは表面再構成で破綻し
> 写実性が落ちた。Splat のままが圧倒的に綺麗なので Splat 方式を採用。
> （メッシュが必要な場合は `scripts/gs_parts_to_mesh.py` で glb/obj を生成可能）

---

## 0. 入力データ

`unity_parts/`（`scripts/recenter_gs_parts.py` の出力）:

| ファイル | 内容 |
|----------|------|
| `eyebrow_eye_right.ply` | 右目 + 右眉 |
| `eyebrow_eye_left.ply`  | 左目 + 左眉 |
| `nose.ply`              | 鼻 |
| `mouth.ply`             | 口・唇 |
| `manifest.json`         | 各パーツの元位置(centroid_offset)・bbox・点数 |

各 `.ply` は **標準 INRIA 3DGS 形式**（`x,y,z,nx,ny,nz,f_dc,f_rest,opacity,scale,rot`）で、
**各パーツの重心が原点**になるよう平行移動済み（＝掴む・回すのに扱いやすいピボット）。
誤割当の浮いたスプラット(floater)は除去済み。

生成し直す場合:
```powershell
# 分割（福笑い構成）
conda run -n facelift python scripts/segment_gaussians.py `
  --ply_path facelift_output/face/gaussians.ply `
  --face_image facelift_output/face/input.png `
  --face_parse_root face-parsing.PyTorch/ `
  --output_dir segmented_fukuwarai/ `
  --camera_json FaceLift/utils_folder/opencv_cameras.json --camera_index 2 `
  --preset fukuwarai --device cuda
# 再センタリング＋クリーニング
conda run -n facelift python scripts/recenter_gs_parts.py `
  --parts_dir segmented_fukuwarai --output_dir unity_parts
```

---

## 1. UnityGaussianSplatting の導入

対象: **Unity 2022.3 LTS 以降**。グラフィックス API は **DX12 または Vulkan**
（Compute Shader 必須。Edit > Project Settings > Player でAPIを確認）。
Built-in / URP / HDRP いずれでも可。

1. パッケージを取得（クローン）:
   ```powershell
   git clone https://github.com/aras-p/UnityGaussianSplatting.git
   ```
2. Unity の **Window > Package Manager** → 左上「+」→ **Add package from disk...**
   → クローンした `UnityGaussianSplatting/package/package.json` を選択。
3. インポートされると **Tools > Gaussian Splats** メニューが増える。

> 手順・メニュー名はバージョンで変わることがある。詳細は同リポジトリの README を参照。

---

## 2. .ply を GaussianSplatAsset に変換

1. **Tools > Gaussian Splats > Create GaussianSplatAsset** を開く。
2. **Input PLY File** に `unity_parts/nose.ply` を指定。
3. **Output Folder** はプロジェクトの `Assets/` 下（例 `Assets/FaceParts/`）。
4. 圧縮品質は **Medium〜High**（数千点と小さいので品質優先で可）。
5. **Create Asset**。→ `nose`（GaussianSplatAsset）が生成される。
6. 残り3パーツも同様に変換。

---

## 3. シーンに配置

1. 空の GameObject を作成（例 `Nose`）。
2. **Add Component → Gaussian Splat Renderer**。
3. **Asset** に手順2で作った `nose` アセットを割り当て。→ 鼻が表示される。
4. 4パーツ分繰り返す。

各パーツは重心が原点なので、GameObject の Transform 原点＝パーツ中心。
掴んで動かす・回すのが自然になる。

### 正しい顔配置に戻す（福笑いの「正解」）
`manifest.json` の `centroid_offset` が各パーツの元の相対位置（共有座標）。
親の空 GameObject（例 `Face`）を作り、その下に各パーツを置いて
`centroid_offset` を local position に入れると元の顔配置に復元できる。

> 注意: UnityGaussianSplatting は取込時に座標系変換（軸の符号反転など）を行う。
> `centroid_offset` は 3DGS 座標系の値なので、そのまま入れると軸が合わない場合がある。
> 1回だけ校正する: 2パーツを並べて正しく見える軸符号の組合せ（例 X反転, Z反転）を
> 確認し、全パーツに同じ符号を適用する。あるいは最初は目視で並べてよい。

| パーツ | centroid_offset (x,y,z) | bbox (x,y,z) |
|--------|--------------------------|--------------|
| nose | (-0.037, -0.572, -0.141) | (0.20, 0.16, 0.27) |
| mouth | (-0.033, -0.546, -0.335) | (0.25, 0.12, 0.09) |
| eyebrow_eye_right | (-0.203, -0.488, -0.006) | (0.11, 0.07, 0.05) |
| eyebrow_eye_left | (0.148, -0.528, 0.112) | (0.24, 0.13, 0.07) |

（値は再生成すると変わるので manifest.json を正とする）

---

## 4. 掴む・当てる（インタラクション用コライダー）

Splat はポリゴンを持たないので、当たり判定は**プリミティブ コライダーを別途付ける**。

1. パーツの GameObject に **Box Collider**（または Sphere/Capsule）を追加。
2. サイズを manifest の `bbox_size` に合わせる（Center は 0 のまま）。
3. VR で掴むなら **XR Interaction Toolkit** を導入し、
   **XR Grab Interactable** + **Rigidbody**（Is Kinematic 推奨）+ 上記 Collider を付ける。

現実の鼻・口の立体物に重ねる用途では、
物理オブジェクト側の位置にパーツ GameObject を追従させる（トラッキング／手動配置）。

---

## 5. 実寸スケール合わせ（AR 重畳向け）

FaceLift のワールド単位は正規化されており、頭部で概ね半径〜1。
現実の立体物に合わせるには `Face` 親をスケールして実寸に合わせる:

- 例: 実際の鼻幅を測り（例 3.5cm）、`nose` の bbox.x(=0.20) と比べて
  スケール係数 = 0.035 / 0.20 ≈ 0.175（Unity 1 unit = 1m 換算）。
- パーツ間の相対位置は `centroid_offset` で保たれているので、
  親を一律スケールすれば配置比率は維持される。

---

## 6. VR / HMD の注意

- **PC 接続（Quest Link / OpenXR）** で PC の GPU 描画なら実用的。
- ステレオ描画は **Single Pass Instanced** で splat が片目にしか出ない場合がある。
  出なければ Project Settings > XR で **Multi Pass** に切替えるか、パッケージを最新化。
- パーツは小さく点数も数百〜数千なので RTX 3080 なら負荷は軽い。

---

## まとめ（この準備で出来ること）
- 各顔パーツを写実的な Gaussian Splat として Unity 内で個別に表示・移動・回転できる。
- manifest の offset で「正しい顔配置」に戻せる（福笑いの正解）。
- コライダーを足せば掴み・物理・現実オブジェクトへの重畳に発展できる。
