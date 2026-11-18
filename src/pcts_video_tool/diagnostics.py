"""Exercise the actual frozen distribution, including models and Qt plugins."""

import json
import os
import sys
import tempfile
import traceback
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile


def self_test(output: str) -> int:
    report = {"ok": False, "frozen": bool(getattr(sys, "frozen", False)), "checks": []}
    try:
        import av
        import cv2
        import numpy as np

        from .analysis import analyze
        from .detection import TextColorTracker, refine_green_start
        from .models import Event, Round, Session, TouchCalibration
        from .recognition import create_hand_detector, create_ocr, fingertip, read_numbers
        from .run_logs import export_diagnostics
        from .touch import ContactTracker, locate_contact
        from .video import inspect_video, read_frame

        image = np.full((120, 650, 3), 255, np.uint8)
        cv2.putText(image, "MD:88.1 TX:-0.7", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
        _, values, _ = read_numbers(create_ocr(), image)
        if values != (88.1, -0.7):
            raise ValueError(f"OCR自检读数不符：{values}")
        report["checks"].append("OCR：本地模型正确读取有符号小数")
        with create_hand_detector() as hand:
            fingertip(hand, np.full((360, 640, 3), 255, np.uint8), 0, [0, 0, 1, 0, 1, 1, 0, 1])
        report["checks"].append("手部模型：CPU推理可运行")
        contact_image = np.full((100, 100, 3), 240, np.uint8)
        contact_image[35:65, 40:80] = (100, 150, 200)
        calibration = TouchCalibration(2, [0.4, 0.2, 0.4, 0.8], [0.6, 0.5])
        tracker = ContactTracker(contact_image, [0, 0, 1, 0, 1, 1, 0, 1], calibration)
        samples = []
        for i, gap in enumerate((12, 6, 0, 0, 0)):
            picture = np.full_like(contact_image, 240)
            picture[35:65, 40 + gap : 80 + gap] = (100, 150, 200)
            samples.append(tracker.measure(i, picture))
        event, _ = locate_contact(samples, 240, latest_press=3)
        if event.frame != 2 or event.state != "review":
            raise ValueError("触摸校准首帧回溯自检失败")
        report["checks"].append("触摸：原图轮廓、参考位置与首帧回溯可用，候选需复核")
        black_text = np.full((120, 360, 3), 230, np.uint8)
        green_text = black_text.copy()
        for picture, color in ((black_text, (0, 0, 0)), (green_text, (0, 180, 0))):
            cv2.putText(picture, "PRESS HERE", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        text_tracker = TextColorTracker(black_text, [0, 0, 1, 0, 1, 1, 0, 1])
        refined = refine_green_start(
            text_tracker,
            lambda frame: black_text if frame < 7 else green_text,
            {"last_black_frame": 3, "first_green_candidate": 12, "confirmed_at": 17},
            240,
        )
        if refined["candidate_frame"] != 7 or refined["confirmed_at"] != 12:
            raise ValueError("green笔画首帧回溯自检失败")
        report["checks"].append("green：字形对齐、亮绿笔画与首帧回溯可用")
        with tempfile.TemporaryDirectory(prefix="pcts-self-test-") as directory:
            video = Path(directory) / "video.nut"
            with av.open(str(video), "w") as container:
                stream = container.add_stream("ffv1", rate=240)
                stream.width, stream.height, stream.pix_fmt = 650, 120, "bgr0"
                for i in range(3):
                    frame = av.VideoFrame.from_ndarray(image, format="bgr24")
                    frame.pts, frame.time_base = i, Fraction(1, 240)
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            info = inspect_video(str(video))
            if len(info.times) != 3 or read_frame(info, 2).shape != image.shape:
                raise ValueError("视频解码自检失败")
            av.Codec("hevc", "r")
            report["checks"].append("视频：解码、精确回看、HEVC解码器可用")
            os.environ["PCTS_LOG_DIR"] = str(Path(directory) / "diagnostics")
            trial = Session(info)
            try:
                analyze(trial)
            except ValueError as error:
                if "框选" not in str(error):
                    raise
            if (
                trial.diagnostic_dir is None
                or not (Path(trial.diagnostic_dir) / "error.txt").is_file()
            ):
                raise ValueError("失败分析的诊断记录未生成")
            bundle = Path(directory) / "diagnostics.zip"
            export_diagnostics(trial, bundle)
            with ZipFile(bundle) as archive:
                if "analysis/error.txt" not in archive.namelist():
                    raise ValueError("诊断包缺少错误记录")
            status = json.loads(
                (Path(trial.diagnostic_dir) / "status.json").read_text(encoding="utf-8")
            )
            if status.get("elapsed_seconds", -1) < 0:
                raise ValueError("诊断记录缺少分析用时")
            report["checks"].append("诊断：失败运行自动记录，日志与任务可导出ZIP")
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            from PySide6.QtWidgets import QApplication

            from .ui import MainWindow

            app = QApplication.instance() or QApplication([])
            report["qt_platform"] = app.platformName()
            window = MainWindow()
            window.accept_session(
                Session(
                    info, rois={"green": [0, 0, 1, 0, 1, 1, 0, 1]}, rounds=[Round(green=Event(1))]
                )
            )
            window.table.selectRow(0)
            window.review_cell(0, 4)
            if window.comparison.frame_indices != [0, 1, 2] or any(
                pane.pixmap.isNull() for pane in window.comparison.panes
            ):
                raise ValueError("三帧对照自检失败")
            window.finish_analysis_clock(65)
            if not window.elapsed_label.text().endswith("00:01:05"):
                raise ValueError("分析用时显示自检失败")
            report["checks"].append("复核：选区三帧对照与分析用时显示可用")
            window.show()
            app.processEvents()
            if window.grab().isNull():
                raise ValueError("界面渲染自检失败")
            window.dirty = False
            window.close()
            report["checks"].append("Qt：主界面、视频预览和离屏渲染可用")
        report["ok"] = True
    except Exception:  # One diagnostic entry boundary must preserve the failure in the report.
        report["error"] = traceback.format_exc()
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["ok"] else 1
