# ============================================================
# rebuild_from_photo.ps1
#   新しい顔写真1枚から、3DGS生成→パーツ分割→テクスチャメッシュ→Unity配置/シーン構築
#   までを一括自動実行し、各段の所要時間を計測して表示する。
#
#   使い方（input\face.jpg に写真を置いてから）:
#     .\windows\rebuild_from_photo.ps1
#   任意:
#     -Photo <path>            入力写真（既定 input\face.jpg）
#     -UnityProject <path>     Unityプロジェクト（既定: リポ同梱 unity\FukuwaraiXR）
#     -Preset <name>           分割プリセット（既定 fukuwarai=目+眉まとめ左右別・鼻・口）
#     -SkipInfer               推論をスキップ（既存 .ply を再利用）
#     -SkipUnity               Unity のメッシュ/シーン構築をスキップ
#
#   ※ Unity 構築を行う場合は、対象プロジェクトを Unity エディタで開いていないこと
#      （プロジェクトがロックされ batchmode が失敗するため）。
#   ※ UTF-8(BOM付き) で保存すること。
# ============================================================
param(
  [string]$Photo = "input\face.jpg",
  [string]$UnityProject = "",
  [string]$Preset = "fukuwarai",
  [switch]$SkipInfer,
  [switch]$SkipUnity
)
$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot
# 既定はリポ同梱の Unity プロジェクト（clone だけで再現できるように相対パス）
if (-not $UnityProject) { $UnityProject = Join-Path $RepoRoot "unity\FukuwaraiXR" }
$EnvName = if ($env:FACELIFT_ENV) { $env:FACELIFT_ENV } else { "facelift" }
$UnityExe = "C:\Program Files\Unity\Hub\Editor\6000.3.19f1\Editor\Unity.exe"
$CamJson = "FaceLift/utils_folder/opencv_cameras.json"

# conda を PATH に
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
  foreach ($c in @("$env:USERPROFILE\miniconda3","$env:USERPROFILE\Anaconda3","$env:LOCALAPPDATA\miniconda3","C:\ProgramData\miniconda3")) {
    if (Test-Path "$c\Scripts\conda.exe") { $env:PATH = "$c;$c\Scripts;$c\Library\bin;$env:PATH"; break }
  }
}
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$timings = [ordered]@{}
function Stage($name, $block) {
  Write-Host "`n==== $name ====" -ForegroundColor Cyan
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  & $block
  $sw.Stop()
  $timings[$name] = $sw.Elapsed
  Write-Host ("---- $name : {0:mm\:ss} ----" -f $sw.Elapsed) -ForegroundColor DarkCyan
}

$swAll = [System.Diagnostics.Stopwatch]::StartNew()

# 1) FaceLift 推論（3DGS）
if (-not $SkipInfer) {
  if (-not (Test-Path $Photo)) { throw "写真がありません: $Photo" }
  New-Item -ItemType Directory -Force facelift_input, facelift_output | Out-Null
  Copy-Item -Force $Photo "facelift_input\face.jpg"
  Stage "1. FaceLift inference (3DGS)" {
    Push-Location FaceLift
    conda run -n $EnvName --no-capture-output python inference.py -i ../facelift_input/ -o ../facelift_output/ --seed 4 --guidance_scale_2D 3.0 --step_2D 50
    Pop-Location
  }
}

$ply = Get-ChildItem -Recurse facelift_output -Filter gaussians.ply -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Desc | Select-Object -First 1
if (-not $ply) { throw "gaussians.ply が見つかりません（推論が必要かも）" }
$faceDir = $ply.Directory.FullName
$texture = Join-Path $faceDir "input.png"
Write-Host "PLY: $($ply.FullName)"
Write-Host "TEX: $texture"

# 2) パーツ分割（目+眉まとめ左右別・鼻・口。dilateで目と眉を連結）
# 古いパーツが混ざらないよう作業ディレクトリを掃除
Remove-Item -Recurse -Force segmented_split, textured_split -ErrorAction SilentlyContinue
Stage "2. Segment parts ($Preset)" {
  conda run -n $EnvName --no-capture-output python scripts/segment_gaussians.py `
    --ply_path $ply.FullName --face_image $texture --face_parse_root "face-parsing.PyTorch/" `
    --output_dir "segmented_split/" --camera_json $CamJson --camera_index 2 `
    --preset $Preset --dilate_px 12 --device cuda
}

# 3) テクスチャメッシュ生成（写真テクスチャ版：obj + meshdata.json + tex.png）
Stage "3. Textured mesh (photo)" {
  conda run -n $EnvName --no-capture-output python scripts/gs_parts_to_textured_mesh.py `
    --parts_dir "segmented_split" --camera_json $CamJson --camera_index 2 `
    --texture $texture --output_dir "textured_split"
}

# 3b) GS 版パーツ（立体保持：重心原点に再センタリング＋scene_config）
Stage "3b. Gaussian Splat parts" {
  conda run -n $EnvName --no-capture-output python scripts/recenter_gs_parts.py `
    --parts_dir "segmented_split" --output_dir "unity_parts"
}

# 4) Unity プロジェクトへコピー（写真テクスチャ版=MeshData / GS版=Source）
Stage "4. Copy parts to Unity project" {
  $meshDst = Join-Path $UnityProject "Assets\FaceParts\MeshData"
  New-Item -ItemType Directory -Force $meshDst | Out-Null
  Get-ChildItem "textured_split" -Directory | ForEach-Object {
    $nm = $_.Name
    $pdst = Join-Path $meshDst $nm; New-Item -ItemType Directory -Force $pdst | Out-Null
    Copy-Item -Force (Join-Path $_.FullName "$nm.meshdata.json") $pdst
    Copy-Item -Force (Join-Path $_.FullName "${nm}_tex.png") $pdst
  }
  $gsDst = Join-Path $UnityProject "Assets\FaceParts\Source"
  New-Item -ItemType Directory -Force $gsDst | Out-Null
  Copy-Item -Force "unity_parts\*.ply" $gsDst
  Copy-Item -Force "unity_parts\scene_config.json" $gsDst
  Write-Host "copied mesh->$meshDst  GS->$gsDst"
}

# 5) Unity でシーン構築（GS版 と 写真テクスチャ版 の両方）
if (-not $SkipUnity) {
  if (-not (Test-Path $UnityExe)) { Write-Warning "Unity 実行体が見つかりません: $UnityExe（Unity段はスキップ）" }
  else {
    Stage "5. Unity build (GS + mesh scenes)" {
      $ulog = Join-Path $env:TEMP "fukuwarai_unity_build.log"
      $p1 = Start-Process -FilePath $UnityExe -Wait -PassThru -NoNewWindow -ArgumentList @(
        "-batchmode","-quit","-projectPath",$UnityProject,
        "-executeMethod","FukuwaraiSetup.BuildAll","-logFile",$ulog)
      if ($p1.ExitCode -ne 0) { Write-Warning "GS scene build 失敗。ログ: $ulog" }
      $ulog2 = Join-Path $env:TEMP "fukuwarai_unity_build2.log"
      $p2 = Start-Process -FilePath $UnityExe -Wait -PassThru -NoNewWindow -ArgumentList @(
        "-batchmode","-quit","-projectPath",$UnityProject,
        "-executeMethod","FukuwaraiMeshBuilder.Build","-logFile",$ulog2)
      if ($p2.ExitCode -ne 0) { Write-Warning "mesh scene build 失敗。ログ: $ulog2" }
    }
  }
}

$swAll.Stop()

Write-Host "`n================ 所要時間 ================" -ForegroundColor Green
foreach ($k in $timings.Keys) { Write-Host ("{0,-38} {1:mm\:ss\.f}" -f $k, $timings[$k]) }
Write-Host ("{0,-38} {1:mm\:ss\.f}" -f "合計", $swAll.Elapsed) -ForegroundColor Green
Write-Host "`n完了。Unity で Assets/Scenes/FukuwaraiFace.unity を確認してください。"
