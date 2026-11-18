@echo off
pushd "%~dp0"
if not exist ".build-venv\Scripts\python.exe" (
  echo Run build-windows.cmd first to prepare the development environment.
  pause
  popd
  exit /b 1
)
".build-venv\Scripts\python.exe" -m pcts_video_tool
if errorlevel 1 pause
popd

