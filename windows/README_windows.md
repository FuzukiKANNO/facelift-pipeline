# Windows (RTX 3070 / Ampere) セットアップ手順

RTX 3070 は Ampere(sm_86) なので、**FaceLift 公式のテスト済み構成**
（torch 2.4.0 + cu124 + xformers 0.0.27.post2）がそのまま使えます。
Blackwell(5090) 特有の問題・パッチは一切不要です。

> ⚠️ **VRAM 注意**: RTX 3070 は 8GB です。FaceLift 推奨は 16GB 以上。
> OOM（メモリ不足）で落ちる可能性があります。実行前に他アプリ・ブラウザ・
> ゲーム等を全て閉じて VRAM を空けてください。それでも落ちる場合は相談を。

---

## 事前に手動で導入するもの（GUI インストーラ）

1. **Miniconda**（Windows 版）
   https://docs.conda.io/en/latest/miniconda.html
2. **Git for Windows**
   https://git-scm.com/download/win
3. **Visual Studio Build Tools 2022**（rasterizer のコンパイルに必須）
   https://visualstudio.microsoft.com/downloads/ → "Build Tools for Visual Studio"
   → インストール時に **「C++ によるデスクトップ開発」** に必ずチェック
4. **NVIDIA ドライバ**（最新。`nvidia-smi` が動く状態）

---

## 手順

### 1. リポジトリ取得
**通常の PowerShell で構いません**（setup が VS の C++ 環境を自動で読み込みます）:
```powershell
cd $HOME
git clone https://github.com/FuzukiKANNO/facelift-pipeline.git
cd facelift-pipeline
```
> Miniconda を「AddToPath 無し」で入れていても、setup/run が自動で PATH を通します。

### 2. セットアップ（環境構築・ビルド・重みDLを自動実行）
```powershell
.\windows\setup_windows.ps1
```
- もし `PowerShell スクリプトの実行がブロックされる` 場合:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```
  を実行してから再度 setup を叩いてください。

### 3. 顔画像を配置
`input\face.jpg` に正面・鮮明な顔写真を置く（ファイル名は face.jpg）。

### 4. 実行（推論→分割→検証）
```powershell
.\windows\run_windows.ps1
```

### 5. 結果確認
- `segmented_output\*.ply` … パーツ別 Gaussian
- `facelift_output\<名前>\turntable.mp4` … FaceLift 本来のレンダリング（品質確認用）
- `.ply` は **SuperSplat**(https://superspl.at/editor) にドラッグ&ドロップで表示

---

## トラブルシューティング

| 症状 | 対処 |
|------|------|
| `cl.exe が見つからない`／ビルド失敗 | VS 2022 に「C++ によるデスクトップ開発」ワークロードが入っているか確認（setup が DevShell を自動ロードします） |
| CUDA out of memory | 他アプリ（Unity/ブラウザ等）を閉じる。それでもダメなら 8GB では厳しい可能性 |
| スクリプト実行がブロック | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| gdown で重みDL失敗 | face-parsing.PyTorch の README から手動DLし `face-parsing.PyTorch\res\cp\79999_iter.pth` に配置 |
| `turntable.mp4` が出来ない (`ffprobe not found`) | `conda install -n facelift -c conda-forge --override-channels ffmpeg -y` |
| seg/verify が `cp932` の UnicodeEncodeError | run_windows.ps1 が `PYTHONUTF8=1` を設定済み。手動実行時も同様に設定 |

> **注意**: `windows\*.ps1` は **UTF-8 (BOM 付き)** で保存してください。BOM 無しだと
> Windows PowerShell 5.1 が ANSI 誤認し、日本語・ヒアストリングが壊れてパースエラーになります。

---

## Linux 版との違い
- 投影軸の修正（横=X / 縦=Z）や分割ロジックは共通（`scripts/` を流用）。
- xformers が正規に使えるので、Linux(5090)で必要だった SDPA 差し替えパッチは
  **適用しません**（公式のアテンション経路をそのまま使用）。
