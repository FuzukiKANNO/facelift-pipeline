@echo off
rem ============================================================
rem  rebuild.bat - one-command launcher (ASCII only for cmd.exe safety)
rem  Put a photo at input\face.jpg, then double-click this file
rem  (or run "rebuild.bat" in cmd/PowerShell).
rem  Runs: 3DGS -> part segmentation -> textured mesh -> Unity build.
rem  NOTE: close the Unity editor for FukuwaraiXR first (project lock).
rem  Options are passed through, e.g.:
rem    rebuild.bat -SkipInfer     (reuse existing .ply)
rem    rebuild.bat -SkipUnity     (skip the Unity step)
rem    rebuild.bat -Photo path\to.jpg
rem ============================================================
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "windows\rebuild_from_photo.ps1" %*
set RC=%ERRORLEVEL%
echo.
echo ==== done (exit %RC%). Press any key to close. ====
pause >nul
exit /b %RC%
