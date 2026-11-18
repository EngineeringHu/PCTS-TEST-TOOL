from __future__ import annotations

import copy
import csv
import json
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

import av
import cv2

from . import __version__
from .detection import FeatureExtractor, TextColorTracker, detect_rounds, refine_green_start
from .marker import TestWindow, find_test_window
from .models import Session, check_source, save_session
from .recognition import (
    MAX_READING_RETRIES,
    choose_reading,
    create_hand_detector,
    create_ocr,
    fingertip,
    read_numbers,
    supplement_reading,
    touch_candidate,
)
from .run_logs import log_directory, logger
from .touch import ContactTracker, TouchSample, locate_contact
from .video import crop, display_image, read_frame, sampling_fps


def analyze(
    session: Session,
    progress=lambda message: None,
    cancelled=lambda: False,
    percentage=lambda value: None,
) -> Session:
    started = perf_counter()
    percentage(0)
    directory = log_directory() / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    directory.mkdir(parents=True)
    session.diagnostic_dir = str(directory)
    save_session(session, directory / "input.session.json")
    (directory / "status.json").write_text(
        json.dumps({"version": __version__, "status": "running"}), encoding="utf-8"
    )
    with (directory / "run.log").open("w", encoding="utf-8") as log:

        def report(message):
            log.write(f"{datetime.now().isoformat(timespec='seconds')} {message}\n")
            log.flush()
            progress(message)

        report(f"开始分析 v{__version__}；诊断目录：{directory}")
        try:
            result = _analyze(session, report, cancelled, directory, percentage)
        except Exception as error:
            (directory / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            (directory / "status.json").write_text(
                json.dumps(
                    {
                        "version": __version__,
                        "status": "cancelled" if isinstance(error, InterruptedError) else "failed",
                        "error": str(error),
                        "elapsed_seconds": round(perf_counter() - started, 3),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            logger.exception("分析未完成 diagnostic_dir=%s", directory)
            raise
        save_session(result, directory / "result.session.json")
        (directory / "status.json").write_text(
            json.dumps(
                {
                    "version": __version__,
                    "status": "complete",
                    "round_count": len(result.rounds),
                    "elapsed_seconds": round(perf_counter() - started, 3),
                    "outcome": "no_rounds"
                    if not result.rounds
                    else ("review" if result.problems() else "ready"),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        logger.info("分析完成 rounds=%s diagnostic_dir=%s", len(result.rounds), directory)
        percentage(100)
        return result


def _analyze(session: Session, progress, cancelled, directory: Path, percentage) -> Session:
    if set(session.rois) != {"flash", "green", "reading", "touch"}:
        raise ValueError("请先框选灯光、变绿文字、MD/TX、触摸观察四个区域")
    check_source(session.video)
    result = copy.deepcopy(session)
    progress("加载本地OCR，检查有效测试片段…")
    ocr, hand = None, None
    result.analysis_notes = []
    contact = None
    if result.touch_calibration is not None:
        contact = ContactTracker(
            read_frame(result.video, result.touch_calibration.frame),
            result.rois["touch"],
            result.touch_calibration,
        )
        for name, picture in (
            ("touch-reference.png", contact.reference),
            ("touch-reference-mask.png", contact.reference_mask * 255),
        ):
            ok, encoded = cv2.imencode(".png", picture)
            if ok:
                (directory / name).write_bytes(encoded.tobytes())
        (directory / "touch-calibration.json").write_text(
            json.dumps(
                {
                    "calibration": asdict(result.touch_calibration),
                    "roi": result.rois["touch"],
                    "reference_front_px": contact.reference_front,
                    "reference_area_px": contact.reference_area,
                    "color_lab": contact.color.tolist(),
                    "color_tolerance": contact.color_tolerance,
                    "meaning": "原图轮廓相对校准位置的像素距离，不是物理间隙毫米值",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        result.analysis_notes.append(
            "press使用原图触摸校准；参考位置、停留与展宽生成候选，仍需复核"
        )
    else:
        result.analysis_notes.append(
            "尚未设置触摸校准，press使用关键点减速候选；可在顶部设置触摸校准"
        )
    try:
        ocr = create_ocr()
    except (ImportError, FileNotFoundError, RuntimeError, ValueError) as error:
        logger.exception("OCR初始化失败")
        result.analysis_notes.append(f"OCR未启用：{error}")
    percentage(5)
    if result.range_mode == "auto":
        with (directory / "marker-checks.jsonl").open("w", encoding="utf-8") as marker_log:

            def record_marker(value):
                marker_log.write(json.dumps(value, ensure_ascii=False) + "\n")
                marker_log.flush()

            window = (
                find_test_window(
                    result.video,
                    result.rois["green"],
                    ocr,
                    progress,
                    cancelled,
                    record_marker,
                    lambda value: percentage(5 + 15 * value // 100),
                )
                if ocr
                else TestWindow(
                    None, None, False, "OCR不可用，无法识别PRESS HERE区间，请手工设置", []
                )
            )
        result.test_start, result.test_end = window.start, window.end
        result.range_confirmed = window.confirmed
    else:
        window = TestWindow(
            result.test_start, result.test_end, True, "使用人工选定的有效测试片段", []
        )
        result.range_confirmed = True
    (directory / "marker.json").write_text(
        json.dumps(asdict(window), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result.analysis_notes.append(window.reason)
    percentage(20)
    if window.start is None or window.end is None:
        result.rounds = []
        progress(window.reason)
        return result
    progress(f"有效测试帧：{window.start}～{window.end}；区间外画面不参与轮次识别")
    # Keep original indices; prefix placeholders are outside the analyzed interval.
    lights, greens, blacks = [0.0] * window.start, [0.0] * window.start, [0.0] * window.start
    extract = FeatureExtractor(result.video.width, result.video.height, result.rois)
    reference_frame = min(
        (
            c["frame"]
            for c in window.checks
            if c["state"] == "present" and window.start <= c["frame"] <= window.end
        ),
        default=window.start,
    )
    tracker = TextColorTracker(read_frame(result.video, reference_frame), result.rois["green"])
    ok, encoded = cv2.imencode(".png", tracker.glyph)
    if ok:
        (directory / "text-template.png").write_bytes(encoded.tobytes())
    (directory / "text-tracking.json").write_text(
        json.dumps(
            {
                "reference_frame": reference_frame,
                "source_roi": result.rois["green"],
                "search_roi": tracker.search_roi,
                "scales": [scale for scale, _ in tracker.templates],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with (directory / "signals.csv").open("w", encoding="utf-8", newline="") as stream:
        signals = csv.writer(stream)
        signals.writerow(
            [
                "frame",
                "timestamp_s",
                "flash_brightness",
                "green_shape_score",
                "black_shape_score",
                "green_ratio",
                "black_ratio",
                "green_x",
                "green_y",
                "green_scale",
                "black_x",
                "black_y",
                "black_scale",
            ]
        )
        with av.open(result.video.path) as container:
            container.streams.video[0].thread_type = "AUTO"
            for index, frame in enumerate(container.decode(video=0)):
                if cancelled():
                    raise InterruptedError("已取消")
                if index > window.end:
                    break
                if index < window.start:
                    continue
                image = display_image(frame)
                light, green_ratio, black_ratio = extract(image)
                green, black = tracker(image)
                lights.append(light)
                greens.append(green)
                blacks.append(black)
                gm, bm = tracker.last_matches["green"], tracker.last_matches["black"]
                signals.writerow(
                    [
                        index,
                        result.video.times[index],
                        light,
                        green,
                        black,
                        green_ratio,
                        black_ratio,
                        *gm["position"],
                        gm["scale"],
                        *bm["position"],
                        bm["scale"],
                    ]
                )
                if index % 120 == 0:
                    percentage(
                        20 + 25 * (index - window.start + 1) // (window.end - window.start + 1)
                    )
                    progress(
                        f"扫描有效片段：{index - window.start + 1}/{window.end - window.start + 1}帧"
                    )
    detected = detect_rounds(
        lights, greens, blacks, sampling_fps(result.video), window.start, window.end + 1
    )
    (directory / "detection.json").write_text(
        json.dumps(detected.diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result.rounds = detected.rounds
    result.analysis_notes.extend(detected.notes)
    percentage(45)
    if not result.rounds:
        return result
    progress("准备原图触摸轮廓与读数…" if contact else "加载手部模型并读取本轮数值…")
    if contact is None:
        # A calibrated local tracker does not depend on the optional hand model.
        try:
            hand = create_hand_detector()
        except (ImportError, FileNotFoundError, RuntimeError, ValueError) as error:
            logger.exception("手部模型初始化失败")
            result.analysis_notes.append(f"手部识别未启用：{error}")
    detection_fps = sampling_fps(result.video)
    ocr_at = {}
    windows = []
    for r, (item, end) in enumerate(zip(result.rounds, detected.green_ends)):
        if item.green.frame is not None:
            start = item.green.frame
            # Read only inside this green segment; never borrow a following round's values.
            for factor in (0.35, 0.65, 0.9):
                frame = min(end - 1, start + int((end - start) * factor))
                ocr_at.setdefault(frame, set()).add(r)
        if item.blink1.frame is not None:
            previous_end = max(
                [window.start]
                + [b for a, b in detected.diagnostics["green_segments"] if a < item.green.frame]
            )
            stop = (
                min(end - 1, item.blink1.frame + max(3, round(detection_fps * 0.05)))
                if contact
                else item.blink1.frame
            )
            windows.append((previous_end, stop, r))
    readings = [[] for _ in result.rounds]
    points = [[] for _ in result.rounds]
    touch_samples = [[] for _ in result.rounds]
    touch_decisions = []
    with (
        (directory / "ocr-checks.jsonl").open("w", encoding="utf-8") as reading_log,
        (directory / "touch-signals.csv").open("w", encoding="utf-8", newline="") as touch_log,
    ):
        touch_writer = csv.DictWriter(
            touch_log, fieldnames=["round", "method", *TouchSample.__dataclass_fields__]
        )
        touch_writer.writeheader()

        def record_reading(round_index, frame, kind, reading):
            text, value, score = reading
            reading_log.write(
                json.dumps(
                    {
                        "round": round_index + 1,
                        "frame": frame,
                        "timestamp_s": result.video.times[frame],
                        "kind": kind,
                        "text": text,
                        "value": value,
                        "score": score,
                        "format_valid": value is not None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            reading_log.flush()

        try:
            with av.open(result.video.path) as container:
                container.streams.video[0].thread_type = "AUTO"
                previous_ms = -1
                for i, frame in enumerate(container.decode(video=0)):
                    if cancelled():
                        raise InterruptedError("已取消")
                    if i > window.end:
                        break
                    if i < window.start:
                        continue
                    if i % 60 == 0:
                        percentage(
                            45 + 30 * (i - window.start + 1) // (window.end - window.start + 1)
                        )
                        progress(
                            f"读取数字与分析触摸动作：{i - window.start + 1}/{window.end - window.start + 1}帧"
                        )
                    active = (
                        [r for start, end, r in windows if start <= i <= end]
                        if (hand or contact)
                        else []
                    )
                    if not active and (not ocr or i not in ocr_at):
                        continue
                    image = display_image(frame)
                    if ocr and i in ocr_at:
                        reading = read_numbers(ocr, crop(image, result.rois["reading"]))
                        for r in ocr_at[i]:
                            readings[r].append((i, *reading))
                            record_reading(r, i, "seed", reading)
                    if active and contact:
                        if any(i == start for start, _, _ in windows):
                            contact.previous_center = contact.reference_center.copy()
                        sample = contact.measure(i, image)
                        for r in active:
                            touch_samples[r].append(sample)
                            touch_writer.writerow(
                                {"round": r + 1, "method": "calibrated_contour", **asdict(sample)}
                            )
                    elif active:
                        # Tracking's monotonic clock never replaces measurement timestamps.
                        t = result.video.times[i]
                        ms = max(
                            previous_ms + 1, round((t if t is not None else i / result.fps) * 1000)
                        )
                        previous_ms = ms
                        point = fingertip(hand, image, ms, result.rois["touch"])
                        if point:
                            for r in active:
                                points[r].append((i, *point))
                        sample = (
                            TouchSample(
                                i,
                                x=point[0] * (result.video.width - 1),
                                y=point[1] * (result.video.height - 1),
                            )
                            if point
                            else TouchSample(i)
                        )
                        for r in active:
                            touch_writer.writerow(
                                {"round": r + 1, "method": "hand_landmark", **asdict(sample)}
                            )
                    if active and i % 60 == 0:
                        touch_log.flush()
        finally:
            if hand:
                hand.close()
        percentage(75)
        neighbor_step = max(1, round(sampling_fps(result.video) / 24))
        for r, item in enumerate(result.rounds):
            if cancelled():
                raise InterruptedError("已取消")
            progress(f"核对MD/TX：第{r + 1}/{len(result.rounds)}轮")

            def read_at(frame, round_index=r):
                reading = read_numbers(
                    ocr, crop(read_frame(result.video, frame), result.rois["reading"])
                )
                record_reading(round_index, frame, "neighbor", reading)
                return reading

            def retry_progress(frame, count, round_index=r):
                progress(
                    f"补读MD/TX：第{round_index + 1}/{len(result.rounds)}轮，"
                    f"原片第{frame}帧（追加{count}/{MAX_READING_RETRIES}帧）"
                )
                percentage(
                    75 + int(20 * (round_index + count / MAX_READING_RETRIES) / len(result.rounds))
                )

            if ocr:
                supplement_reading(
                    item,
                    readings[r],
                    item.green.frame,
                    detected.green_ends[r],
                    read_at,
                    cancelled,
                    retry_progress,
                    neighbor_step,
                )
            else:
                choose_reading(item, readings[r])
            if contact:
                item.press, decision = locate_contact(
                    touch_samples[r], detection_fps, item.blink1.frame
                )
            else:
                item.press = touch_candidate(points[r])
                decision = {"method": "hand_landmark_deceleration", "sample_count": len(points[r])}
            touch_decisions.append({"round": r + 1, "event": asdict(item.press), **decision})
            percentage(75 + 20 * (r + 1) // len(result.rounds))
    (directory / "touch-detection.json").write_text(
        json.dumps(touch_decisions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Refine only after the original sampling passes: other fields keep their evidence.
    proofs = {row["first_green_candidate"]: row for row in detected.diagnostics["green_evidence"]}
    detected.diagnostics["coarse_green_segments"] = list(detected.diagnostics["green_segments"])
    refinements = []
    with (directory / "green-refinement.jsonl").open("w", encoding="utf-8") as refinement_log:
        for r, item in enumerate(result.rounds):
            if cancelled():
                raise InterruptedError("已取消")
            coarse = item.green.frame
            progress(f"细化green首帧：第{r + 1}/{len(result.rounds)}轮")

            def record(value, round_index=r):
                refinement_log.write(
                    json.dumps({"round": round_index + 1, **value}, ensure_ascii=False) + "\n"
                )
                refinement_log.flush()

            def refine_progress(frame, round_index=r):
                progress(
                    f"细化green首帧：第{round_index + 1}/{len(result.rounds)}轮，原片第{frame}帧"
                )

            if item.green.state == "review":
                refinement = refine_green_start(
                    tracker,
                    lambda frame: read_frame(result.video, frame),
                    proofs[coarse],
                    detection_fps,
                    window.start,
                    cancelled,
                    refine_progress,
                    record,
                )
            else:
                refinement = {
                    "coarse_frame": coarse,
                    "candidate_frame": None,
                    "reason": "原候选已自动接受，无需回溯",
                }
            candidate = refinement["candidate_frame"]
            earliest = item.blink2.frame if item.blink2.frame is not None else item.blink1.frame
            if candidate is not None and candidate < earliest:
                refinement["reason"] = "笔画候选早于本轮灭灯，保留原候选并复核顺序"
                item.green.reason += "；笔画回溯与灭灯顺序冲突，需人工确认"
            elif candidate is not None:
                item.green.frame = candidate
                item.green.state = "review"
                item.green.reason = (
                    f"笔画颜色回溯：首绿候选由{coarse}提前至{candidate}帧，"
                    f"在{refinement['confirmed_at']}帧确认；过渡边界仍需复核"
                )
                proofs[coarse]["refined_frame"] = candidate
                proofs[coarse]["refinement_confirmed_at"] = refinement["confirmed_at"]
                detected.diagnostics["green_segments"] = [
                    (candidate if start == coarse else start, end)
                    for start, end in detected.diagnostics["green_segments"]
                ]
            refinement["selected_frame"] = item.green.frame
            refinement["applied"] = item.green.frame != coarse
            refinements.append({"round": r + 1, **refinement})
            detected.diagnostics["round_candidates"][r]["coarse_green"] = coarse
            detected.diagnostics["round_candidates"][r]["green"] = item.green.frame
            record({"kind": "result", **refinement})
            percentage(95 + 4 * (r + 1) // len(result.rounds))
    detected.diagnostics["green_refinements"] = refinements
    (directory / "detection.json").write_text(
        json.dumps(detected.diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    corrected = sum(row["applied"] for row in refinements)
    if corrected:
        result.analysis_notes.append(f"green笔画颜色回溯调整了{corrected}轮首帧，过渡边界保留复核")
    progress(f"分析完成：找到{len(result.rounds)}轮，疑难项可逐帧修正")
    return result
