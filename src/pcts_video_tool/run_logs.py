from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict
from logging.handlers import RotatingFileHandler
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from . import __version__
from .models import Session

logger = logging.getLogger("pcts_video_tool")


def log_directory() -> Path:
    if os.environ.get("PCTS_LOG_DIR"):
        return Path(os.environ["PCTS_LOG_DIR"])
    if sys.platform == "win32":
        return Path(os.environ["LOCALAPPDATA"]) / "PCTSVideoTool" / "diagnostics"
    return Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "pcts-video-tool"


def configure_logging() -> None:
    directory = log_directory()
    directory.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        directory / "application.log", maxBytes=5_000_000, backupCount=2, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info("程序启动 version=%s python=%s", __version__, sys.version.split()[0])


def export_diagnostics(session: Session | None, path: Path) -> None:
    root = log_directory()
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        application_log = root / "application.log"
        if application_log.exists():
            archive.write(application_log, "application.log")
        if session is not None:
            archive.writestr(
                "current.session.json",
                json.dumps(
                    {"format": 4, **asdict(session)}, ensure_ascii=False, indent=2, allow_nan=False
                ),
            )
            if session.diagnostic_dir:
                # Only export this application's known diagnostic files, never the source video.
                directory = root / Path(session.diagnostic_dir).name
                for name in (
                    "input.session.json",
                    "result.session.json",
                    "signals.csv",
                    "detection.json",
                    "green-refinement.jsonl",
                    "text-template.png",
                    "text-tracking.json",
                    "marker.json",
                    "marker-checks.jsonl",
                    "ocr-checks.jsonl",
                    "touch-calibration.json",
                    "touch-reference.png",
                    "touch-reference-mask.png",
                    "touch-signals.csv",
                    "touch-detection.json",
                    "run.log",
                    "error.txt",
                    "status.json",
                ):
                    source = directory / name
                    if source.is_file():
                        archive.write(source, "analysis/" + name)
