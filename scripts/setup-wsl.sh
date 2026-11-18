#!/usr/bin/env bash
set -euo pipefail
PCTS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PCTS_ROOT"
python3 -m venv --without-pip .venv
python3 -m pip --python .venv/bin/python install -e '.[dev]'
.venv/bin/python scripts/prepare_models.py
mkdir -p .runtime/packages
(
    cd .runtime/packages
    apt-get download libegl1 libopengl0
)
for package in .runtime/packages/*.deb; do
    dpkg-deb -x "$package" .runtime
done
echo '准备完成。Windows 双击 start-windows.vbs，或在 WSL 执行 bash scripts/run-wsl.sh。'
