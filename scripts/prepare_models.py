"""Explicit installation-time model download; application never downloads models."""

import hashlib
import urllib.request
from pathlib import Path

URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
EXPECTED_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
root = Path(__file__).resolve().parents[1]
directory = root / "models"
directory.mkdir(exist_ok=True)
target = directory / "hand_landmarker.task"
if not target.exists():
    temporary = target.with_suffix(".download")
    urllib.request.urlretrieve(URL, temporary)
    temporary.replace(target)
digest = hashlib.sha256(target.read_bytes()).hexdigest()
if digest != EXPECTED_SHA256:
    raise ValueError("手部模型校验不匹配，请检查下载文件")
print(f"Model: {target}\nSHA256: {digest}")
