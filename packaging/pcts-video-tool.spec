from pathlib import Path
from importlib.util import find_spec
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

root = Path(SPECPATH).parent
datas = collect_data_files("rapidocr") + [(str(root / "models" / "hand_landmarker.task"), "models")]
datas += [(str(root / "docs" / "AI_performance_test.xlsx"), "templates")]
binaries = collect_dynamic_libs("mediapipe")
# Wine cannot always query QtLibraryInfo; collect the actual Windows GUI plugins explicitly.
qt_root = Path(find_spec("PySide6").origin).parent
for category in ("platforms", "styles"):
    binaries += [(str(path), f"PySide6/plugins/{category}")
                 for path in (qt_root / "plugins" / category).glob("*.dll")]
a = Analysis(
    [str(root / "packaging" / "launcher.py")],
    pathex=[str(root / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=collect_submodules("rapidocr.inference_engine.onnxruntime"),
    excludes=["tkinter", "torch", "paddle", "tensorflow"],
)
# Preserve the package's spelling on case-sensitive WSL shares as well as NTFS.
a.binaries = [("shapely.libs" + name[len("Shapely.libs"):]
               if name.startswith("Shapely.libs") else name, source, kind)
              for name, source, kind in a.binaries]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="PCTSVideoTool", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="PCTSVideoTool")
standalone = EXE(pyz, a.scripts, a.binaries, a.datas, [], name="PCTSVideoTool", console=False)
