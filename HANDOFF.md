# 引継ぎ書 — FaceLift 顔パーツ別 3DGS 分割パイプライン

別PC（RTX 3070 / Windows）で作業を続けるための引継ぎ書です。
この1枚と、リポジトリ内のスクリプトだけで作業を再開できます。

> **新PCで Claude Code を使う場合**: このリポジトリを clone したフォルダで
> Claude Code を起動し、「HANDOFF.md を読んで作業を続けたい」と伝えてください。
> 会話履歴が無くても、この文書で文脈を復元できます。

---

## 1. 目的（ゴール）

1枚の顔写真から:
1. **FaceLift**（ICCV 2025）で 3D Gaussian Splatting（`.ply`）を生成
2. **BiSeNet** で顔を部位パース（目・鼻・口・肌・髪…）
3. 各 Gaussian を 2D 投影してパース結果と照合し、**パーツ別 `.ply`** に分割
4. 最終的に Unity（UnityGaussianSplatting）でパーツごとに扱う

出力パーツ: `eye_left / eye_right / eyebrow / nose / mouth / skin_cheek / ear / hair / other`

---

## 2. これまでの経緯（重要）

最初は **RTX 5090（Linux）** で構築を進めたが、5090 は Blackwell(sm_120) で
新しすぎて、以下の連続した問題に対処した:

| 問題 | 対処 |
|------|------|
| torch 2.4.0+cu124 が Blackwell 非対応 | torch **2.11.0+cu128** に変更 |
| conda create が ToS で失敗 | conda-forge チャンネル + ToS 承認 |
| rasterizer が `std::uintptr_t` で失敗 | `<cstdint>` を追加パッチ |
| rasterizer が build isolation で失敗 | `--no-build-isolation` |
| xformers が torch2.11/sm_120 非対応 | **SDPA に差し替え**（5090 では推奨） |
| rembg が onnxruntime 不足 | onnxruntime 追加 |
| 拡散が OOM（22GiB確保） | SDPA(AttnProcessor2_0) + expandable_segments |
| 分割パーツが細長い | 投影軸を **横=X / 縦=Z(天地反転)** に修正 |

### ⚠️ 未解決の問題（5090 環境）
FaceLift の出力品質が低い（顔が「ぐちゃっと」崩れる）。切り分けの結果:
- 入力→6視点マルチビュー（拡散）まで**は正常**（`multiview.png` は綺麗で一貫）
- **3D化（GS-LRM）の段だけ崩れる**
- 原因は「torch 2.11 / Blackwell 環境の数値問題」の疑いが濃厚
  （SDPA 差し替えは 5090 で推奨手法なので、それ自体は原因ではないと判断）

→ **この品質問題を回避するために、Ampere の RTX 3070 へ移行する**のが現在の方針。

---

## 3. なぜ RTX 3070（Ampere）に移行するのか

- 3070 は **Ampere(sm_86)** で、FaceLift 公式のテスト済み構成
  （**torch 2.4.0 + cu124 + xformers 0.0.27.post2**）が**そのまま動く**。
- 上記 5090 で必要だったパッチ（SDPA 差し替え等）は**一切不要**。
- したがって **品質のぐちゃっと問題も解消される見込み**（作者の想定環境そのもの）。

### ⚠️ 唯一の懸念: VRAM
RTX 3070 は **8GB**。FaceLift 推奨は 16GB 以上、5090 実行時は 20GB 超使用。
**8GB では OOM で落ちる可能性が現実的にある。** 実行前に他アプリを全て閉じる。
それでも落ちる場合は、この文書末尾「困ったとき」を参照。

---

## 4. 新PC（Windows / RTX 3070）での手順

詳細は [windows/README_windows.md](windows/README_windows.md)。要約:

### 事前準備（GUI 手動インストール）
1. Miniconda
2. Git for Windows
3. **Visual Studio Build Tools 2022**（「C++ によるデスクトップ開発」ワークロード）
4. NVIDIA ドライバ（`nvidia-smi` が動く）

### 実行
「**Developer PowerShell for VS 2022**」を起動して:
```powershell
cd $HOME
git clone https://github.com/FuzukiKANNO/facelift-pipeline.git
cd facelift-pipeline
.\windows\setup_windows.ps1        # 環境構築・ビルド・重みDL（自動）
# input\face.jpg に顔画像を置く
.\windows\run_windows.ps1          # 推論→分割→検証
```

### 成功の確認ポイント
- setup 段階2: `RTX 3070` / capability `(8, 6)` と表示される
- setup 段階5: `rasterizer import OK`
- run 後: `facelift_output\<名前>\turntable.mp4` が**崩れず綺麗**か
- `segmented_output\*.ply` を SuperSplat(https://superspl.at/editor) で確認

---

## 5. リポジトリ内ファイルの役割

| ファイル | 用途 |
|----------|------|
| `windows/setup_windows.ps1` | **Windows(3070) 用**セットアップ（これを使う） |
| `windows/run_windows.ps1` | **Windows 用**実行（推論→分割→検証） |
| `windows/README_windows.md` | Windows 手順の詳細 |
| `scripts/segment_gaussians.py` | Gaussian をパーツ別に分割（投影軸修正済み: 横X/縦Z） |
| `scripts/verify_ply.py` | 出力 `.ply` の検証 |
| `scripts/diagnose_projection.py` | 座標系診断（XY/XZ/YZ 投影画像を出力） |
| `setup_blackwell.sh` / `build_deps.sh` / `run_pipeline.sh` | **Linux(5090) 用**（今回は使わない。5090 の記録として保持） |

> Windows の新規 clone では FaceLift/utils_transformer.py は**未パッチ**（＝xformers 正規経路）。
> これが 5090 環境との決定的な違いで、品質改善が期待できる理由。

---

## 6. 分割の仕組み（要点）

1. FaceLift が 3D の粒（Gaussian）を生成
2. BiSeNet が写真を部位ごとに塗り分け（2D ラベルマップ）
3. 各 3D 粒を写真面に投影し、落ちたピクセルの部位ラベルを付与
4. ラベルごとに `.ply` を分けて保存

投影軸は **横=X, 縦=Z（Z 反転で頭が上）** が正しい（FaceLift 出力は顔が XZ 平面・Y が奥行き）。
左右の目が入れ替わる場合は `segment_gaussians.py` に `--flip_h` を付ける。

---

## 7. 困ったとき

| 症状 | 対処 |
|------|------|
| PowerShell 実行がブロック | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| `cl.exe が無い`/ビルド失敗 | 「Developer PowerShell for VS 2022」から実行しているか、VS Build Tools の C++ が入っているか確認 |
| CUDA out of memory（8GB の壁） | 他アプリ全終了。改善しなければ、`--step_2D` を下げる等を検討（要相談） |
| gdown 失敗 | face-parsing.PyTorch README から手動DL → `face-parsing.PyTorch\res\cp\79999_iter.pth` |
| 品質がまだ低い | `facelift_output\<名前>\multiview.png` を見て 6 視点が一貫しているか確認。一貫していれば入力/seed の問題、崩れていれば拡散側の問題 |

### ログ共有の方法（Claude に見せるとき）
長いエラーはコピペしづらいので、ファイルを匿名アップロードして URL を渡すと早い:
```powershell
# 例: ビルドログを共有
Get-Content _build\raster_build.log -Raw | Out-File -Encoding utf8 log.txt
# log.txt を https://0x0.st などにアップロードするか、内容を貼る
```

---

## 8. 現在のステータス（2026-07-21 更新）

- [x] Linux/5090 環境構築（動くが品質問題あり → 保留）
- [x] Windows 用スクリプト作成（`windows/`）
- [x] Windows/Ampere で setup 実行（**実機は RTX 3080 Laptop 8GB / sm_86**）
- [x] Windows/Ampere で推論 → **品質確認：崩れ問題は解消**（`output.png`/`turntable.mp4` とも綺麗）
- [x] パーツ分割 → 検証OK（合計 162,606 Gaussians が元PLYと一致）
- [ ] SuperSplat / Unity(UnityGaussianSplatting) への取り込み

### ✅ 移行結果（結論）
Ampere + 公式スタック(torch2.4/cu124/xformers) で **5090 の「3D化で崩れる」問題は解消**。
HANDOFF の仮説（5090/Blackwell/torch2.11 の数値問題が原因）が裏付けられた。

### セットアップで実際に踏んだ問題と対処（`windows/` に反映済み）
| 問題 | 対処 |
|------|------|
| `.ps1` が BOM 無し UTF-8 で PS5.1 がパース失敗 | **UTF-8(BOM付き)** で保存 |
| conda が PATH に無い | setup/run 冒頭で Miniconda を自動検出し PATH 追加 |
| `Developer PowerShell` 以外だと cl.exe が無い | setup が VS DevShell を自動ロード（通常の PowerShell で可） |
| `conda run python -c "複数行"` がパッチ失敗 | パッチを `windows/patch_rasterizer.py` に分離 |
| リンクで `cudart.lib` が見つからず失敗 | conda の `<env>\Library\lib` を `LIB` に追加 |
| `ffprobe not found`（turntable未生成） | `ffmpeg` を `-c conda-forge --override-channels` で導入 |
| seg/verify が `cp932` で UnicodeEncodeError | run が `PYTHONUTF8=1` を設定 |

**次にやること**: `segmented_output\*.ply` を SuperSplat(https://superspl.at/editor) で確認し、
Unity(UnityGaussianSplatting) へパーツ別に取り込む。
