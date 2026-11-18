#!/usr/bin/env bash
set -euo pipefail
PCTS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PCTS_ROOT"
if [[ ! -x .venv/bin/python ]]; then
    echo '请先在 WSL 运行 bash scripts/setup-wsl.sh。' >&2
    exit 1
fi
export LD_LIBRARY_PATH="$PCTS_ROOT/.runtime/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-wayland}"
if [[ "$QT_QPA_PLATFORM" == wayland && ! -S "${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/${WAYLAND_DISPLAY:-wayland-0}" ]]; then
    export WAYLAND_DISPLAY=/mnt/wslg/runtime-dir/wayland-0
fi
exec .venv/bin/python -m pcts_video_tool "$@"
