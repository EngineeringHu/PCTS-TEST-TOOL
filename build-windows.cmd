@echo off
pushd "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Install Python 3.13 x64 with the Python Launcher first.
  pause
  popd
  exit /b 1
)
py -3.13 scripts\build_windows.py
if errorlevel 1 (
  echo Build failed. See build\build.log and dist\windows-self-test.json if present.
) else (
  echo Ready: dist\PCTSVideoTool-windows-x64.zip
)
pause
popd
