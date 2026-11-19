"""Exercise the actual frozen distribution, including models and Qt plugins."""

import json
import os
import sys
import tempfile
import traceback
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile

from . import __version__


def self_test(output: str) -> int:
    report = {
        "ok": False,
        "version": __version__,
        "frozen": bool(getattr(sys, "frozen", False)),
        "checks": [],
    }
    try:
        import av
        import cv2
        import numpy as np

        from .analysis import analyze
        from .detection import TextColorTracker, refine_green_start
        from .exporting import export_report
        from .models import Event, GreenRecommendation, Round, Session, TouchCalibration
        from .motion import MotionTracker, automatic_finger_samples, locate_motion_press
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
        with create_hand_detector(image_mode=True) as hand:
            if automatic_finger_samples(
                hand, np.full((360, 640, 3), 255, np.uint8), [0, 0, 1, 0, 1, 1, 0, 1]
            ):
                raise ValueError("自动指腹自检错误：空白画面生成取样")
        report["checks"].append("自动取样：IMAGE手部模型可运行，空白画面不生成指腹")
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
        motion_images = []
        for i, gap in enumerate([12, 6] + [0] * 8):
            picture = np.full_like(contact_image, 240)
            picture[35:65, 40 + gap : 80 + gap] = (100, 150, 200)
            motion_images.append((i, picture))
        finger = MotionTracker(contact_image, [0, 0, 1, 0, 1, 1, 0, 1], [0.6, 0.5])
        event, _, _ = locate_motion_press(finger.trace(motion_images, 4), 240, 2, finger.origin)
        if event.frame != 2 or event.state != "review":
            raise ValueError("自动方向的运动转折自检失败")
        report["checks"].append("自动触摸：无参考线、双向追踪、后帧确认并回溯到运动转折")
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
            template_result = export_report(
                Session(
                    info,
                    rounds=[
                        Round(
                            press=Event(0),
                            blink1=Event(1),
                            blink2=Event(1),
                            green=Event(2),
                            md=12.5,
                            tx=-0.5,
                        )
                    ],
                ),
                Path(directory) / "template-export",
                evidence=False,
            )
            from xml.etree import ElementTree as ET

            with ZipFile(template_result) as archive:
                ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
                relationship = (
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
                )
                workbook = ET.fromstring(archive.read("xl/workbook.xml"))
                worksheet = next(
                    s for s in workbook.find(ns + "sheets") if s.get("name") == "T1 (2)"
                )
                relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
                location = next(
                    r.get("Target")
                    for r in relationships
                    if r.get("Id") == worksheet.get(relationship)
                )
                location = location.lstrip("/") if location.startswith("/") else "xl/" + location
                cells = {
                    c.get("r"): c for c in ET.fromstring(archive.read(location)).iter(ns + "c")
                }
                if (
                    cells["A6"].find(ns + "v").text != "0"
                    or cells["G6"].find(ns + "v").text != "2"
                    or abs(float(cells["O6"].find(ns + "v").text) - 1000 / 120) > 1e-8
                ):
                    raise ValueError("模板Excel填写或耗时缓存错误")
            report["checks"].append("Excel模板：内置工作簿可用，T1 (2)字段映射与耗时缓存正确")
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
            from PySide6.QtCore import Qt, QTimer
            from PySide6.QtTest import QTest
            from PySide6.QtWidgets import QApplication

            from .ui import MainWindow

            app = QApplication.instance() or QApplication([])
            report["qt_platform"] = app.platformName()
            window = MainWindow()
            window.accept_session(
                Session(
                    info,
                    rois={"green": [0, 0, 1, 0, 1, 1, 0, 1]},
                    rounds=[Round(green=Event(1), blink1=Event(2))],
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
            window.comparison.back_button.click()
            window.review_cell(0, 2)
            if window.video_views.currentWidget() is not window.canvas or window.current_frame != 2:
                raise ValueError("单幅全画面模式未跨单元格保留")
            window.review_cell(0, 1)
            if window.current_frame != 0 or window.session.rounds[0].press.frame is not None:
                raise ValueError("缺失press辅助浏览起点或只读状态错误")
            window.step(1)
            window.review_layout.setCurrentIndex(0)
            if window.comparison.frame_indices != [0, 1, 2]:
                raise ValueError("切回三帧模式未保留浏览帧")
            report["checks"].append("复核导航：显示模式跨单元格保留，缺失press辅助定位不写入测量值")
            reader = window.frame_reader
            if (
                not np.array_equal(reader.read(2), read_frame(info, 2))
                or reader.cache_bytes > reader.max_bytes
            ):
                raise ValueError("浏览帧缓存的像素或容量检查失败")
            window.show()
            app.processEvents()
            if window.grab().isNull():
                raise ValueError("界面渲染自检失败")
            window.dirty = False
            window.close()
            if reader.container is not None or reader.cache_bytes:
                raise ValueError("关闭窗口后浏览解码器未释放")
            report["checks"].append("浏览性能：复用原帧缓存，像素保持一致，关闭后释放读取器")
            report["checks"].append("Qt：主界面、视频预览和离屏渲染可用")
            from .region_frames import prepare_region_session

            selection_video = Path(directory) / "selection.nut"
            with av.open(str(selection_video), "w") as container:
                stream = container.add_stream("ffv1", rate=240)
                stream.width, stream.height, stream.pix_fmt = 650, 240, "bgr0"
                for i in range(12):
                    picture = np.full((240, 650, 3), 240, np.uint8)
                    cv2.putText(
                        picture,
                        "PRESS HERE",
                        (50, 175),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.4,
                        (0, 0, 0),
                        3,
                    )
                    if i >= 2:
                        cv2.putText(
                            picture,
                            "MD:88.1 TX:-0.7",
                            (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.4,
                            (0, 160, 0),
                            3,
                        )
                    frame = av.VideoFrame.from_ndarray(picture, format="bgr24")
                    frame.pts, frame.time_base = i, Fraction(1, 240)
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            selection, previews = prepare_region_session(
                str(selection_video), lambda message: None, lambda: False
            )
            if set(previews) != {"flash", "green", "reading"} or previews["reading"][0] < 2:
                raise ValueError("导入辅助选帧自检失败")
            for index, picture in previews.values():
                if not np.array_equal(picture, read_frame(selection.video, index)):
                    raise ValueError("辅助帧与原片像素不一致")
            report["checks"].append("辅助选帧：导入时识别文字与读数，缓存原片画面")
            window = MainWindow()
            window.accept_session(selection, previews)
            window.dirty = False
            # Supply a frame to exercise the touch selection control without inventing a hand.
            window.region_previews["touch"] = previews["flash"]
            for name in ("flash", "green", "reading", "touch"):
                window.region_combo.setCurrentIndex(window.region_combo.findData(name))
                expected_shape = "rectangle" if name == "flash" else "quad"
                if (
                    window.current_frame != previews[name][0]
                    or window.canvas.shape != expected_shape
                    or window.dirty
                    or selection.rois
                    or selection.rounds
                ):
                    raise ValueError(f"{name}辅助框选入口自检失败")
            window.close()
            report["checks"].append("辅助框选：四类区域跳帧、默认形状与浏览状态检查通过")
            selection.rounds = [Round(press=Event(0), blink1=Event(1, "auto"), md=12.5, tx=-0.7)]
            window = MainWindow()
            window.accept_session(selection)
            window.show()
            window.table.setCurrentCell(0, 1)
            window.show_frame(2)
            app.processEvents()
            from .ui import CELL_COLORS

            if (
                window.table.item(0, 1).background().color().name() != CELL_COLORS["review"]
                or window.table.item(0, 2).background().color().name() != CELL_COLORS["accepted"]
                or window.table.item(0, 3).background().color().name() != CELL_COLORS["missing"]
            ):
                raise ValueError("单元格状态配色自检失败")

            def confirm_menu():
                menu = app.activePopupWidget()
                menu.setActiveAction(menu.actions()[-1])
                QTest.keyClick(menu, Qt.Key_Return)

            QTimer.singleShot(0, confirm_menu)
            window.cell_menu(window.table.visualItemRect(window.table.item(0, 1)).center())
            if selection.rounds[0].press != Event(2, "manual", "人工确认当前帧"):
                raise ValueError("右键确认当前帧自检失败")
            window.table.setCurrentCell(0, 5)
            window.table.item(0, 5).setText("-1.2")
            window.confirm_cell(0, 5, window.current_frame)
            item = selection.rounds[0]
            if item.md != -1.2 or item.reading_states != {"md": "manual", "tx": "review"}:
                raise ValueError("单项读数编辑与独立确认自检失败")
            original = Event(10, "auto", "原始green")
            item.blink2 = Event(2, "auto")
            item.green = original
            item.green_recommendation = GreenRecommendation(3, original)
            window.refresh_table()
            window.preview_green.click()
            if window.current_frame != 3 or item.green != original:
                raise ValueError("查看green推荐帧时测量值或浏览帧异常")
            window.apply_green.click()
            if item.green.frame != 3 or item.green.state != "manual":
                raise ValueError("采用green推荐帧失败")
            window.apply_green.click()
            if item.green != original:
                raise ValueError("恢复green原值与状态失败")
            report["checks"].append("green推荐：查看、明确采用和恢复原帧及确认状态")
            window.dirty = False
            window.close()
            report["checks"].append("单元格复核：状态配色、右键确认当前帧、MD/TX独立编辑与确认")
        report["ok"] = True
    except Exception:  # One diagnostic entry boundary must preserve the failure in the report.
        report["error"] = traceback.format_exc()
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["ok"] else 1
