"""One-click native Windows build using the tested Python 3.13 dependency set."""

import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main() -> int:
    if sys.platform != "win32" or sys.version_info[:2] != (3, 13) or struct.calcsize("P") != 8:
        print("构建需要 Windows 和 Python 3.13 x64；最终使用者不需要Python或WSL。")
        return 1
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    (root / "build").mkdir(exist_ok=True)
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONUTF8="1")
    with (root / "build" / "build.log").open("w", encoding="utf-8") as log:

        def run(arguments):
            command = [str(value) for value in arguments]
            print("\n执行：", subprocess.list2cmdline(command), flush=True)
            log.write("\n" + subprocess.list2cmdline(command) + "\n")
            with subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=environment,
            ) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)

        python = root / ".build-venv" / "Scripts" / "python.exe"
        run([sys.executable, "-m", "venv", root / ".build-venv"])
        run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "-r",
                root / "requirements-windows.lock",
            ]
        )
        run([python, "-m", "pip", "install", "--no-deps", "-e", root])
        run([python, root / "scripts" / "prepare_models.py"])
        run([python, "-m", "pytest", "-q"])
        run(
            [
                python,
                "-m",
                "PyInstaller",
                "--noconfirm",
                root / "packaging" / "pcts-video-tool.spec",
            ]
        )
        executable = root / "dist" / "PCTSVideoTool.exe"
        report_path = root / "dist" / "windows-self-test.json"
        run([executable, "--self-test", report_path])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not report.get("ok") or not report.get("frozen"):
            raise RuntimeError(f"冻结包自检未通过，请查看 {report_path}")
        shutil.copy2(root / "README.md", executable.parent / "README.md")
        archive = root / "dist" / "PCTSVideoTool-windows-x64.zip"
        with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
            bundle.write(executable, "PCTSVideoTool.exe")
            bundle.write(root / "README.md", "README.md")
        print(f"\n构建及自检通过：{archive}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, subprocess.CalledProcessError, RuntimeError, ValueError) as error:
        print(
            f"\n构建未完成：{error}\n请检查 build/build.log。未通过自检的EXE不可作为正式产物分发。"
        )
        sys.exit(1)
