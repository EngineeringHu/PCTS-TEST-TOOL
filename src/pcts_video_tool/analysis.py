from __future__ import annotations

import copy
import csv
import json
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

import av
import cv2

from . import __version__
from .detection import FeatureExtractor, TextColorTracker, detect_rounds, refine_green_start
from .gray_recovery import ManualRoundMismatch, recover_gray_rounds, retain_manual_fields
from .green_recommendation import MIN_LEAD_FRAMES, GreenOnsetProbe, interior_glyph
from .marker import TestWindow, find_test_window
from .models import Event, GreenRecommendation, Session, check_source, save_session
from .motion import locate_motion_press, motion_preview, prepare_motion_tracker
from .recognition import (
    MAX_READING_RETRIES,
    choose_reading,
    create_hand_detector,
    create_ocr,
    read_numbers,
    supplement_reading,
)
from .run_logs import log_directory, logger
from .touch import ContactTracker, TouchSample, locate_contact
from .video import FrameReader, crop, display_image, read_frame, sampling_fps


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
    previous_directory = session.diagnostic_dir
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
            confirmed = {r.green.frame for r in session.rounds if r.green.state == "manual"}
            retained = {r.green.frame for r in result.rounds if r.green.state == "manual"}
            if confirmed - retained:
                session.diagnostic_dir = previous_directory
                raise ValueError(
                    "本次轮次与已人工确认的 green 无法逐轮对应，已保留原任务。可重新导入原片建立独立分析。"
                )
        except Exception as error:
            if isinstance(error, ManualRoundMismatch):
                session.diagnostic_dir = previous_directory
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


def _scan_signals(video, window, extract, tracker, cancelled):
    def measure(image):
        return extract(image), tracker.match(image)

    # Decode in presentation order. Only independent pixel measurements run in parallel.
    with av.open(video.path) as container, ThreadPoolExecutor(max_workers=2) as pool:
        container.streams.video[0].thread_type = "AUTO"
        pending = deque()
        for index, frame in enumerate(container.decode(video=0)):
            if cancelled():
                raise InterruptedError("已取消")
            if index > window.end:
                break
            if index >= window.start:
                pending.append((index, pool.submit(measure, display_image(frame))))
                if len(pending) >= 3:
                    first, job = pending.popleft()
                    yield first, *job.result()
        for index, job in pending:
            yield index, *job.result()


def _analyze(session: Session, progress, cancelled, directory: Path, percentage) -> Session:
    if set(session.rois) != {"flash", "green", "reading", "touch"}:
        raise ValueError("请先框选灯光、变绿文字、MD/TX、触摸观察四个区域")
    check_source(session.video)
    result = copy.deepcopy(session)
    progress("加载本地OCR，检查有效测试片段…")
    ocr, hand = None, None
    result.analysis_notes = []
    contact = None
    if result.press_mode == "reference":
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
            "press使用自动方向与连续指腹轨迹，无需参考线；运动转折候选仍需复核"
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
    black_poses = []
    extract = FeatureExtractor(result.video.width, result.video.height, result.rois)
    reference_frame = min(
        (
            c["frame"]
            for c in window.checks
            if c["state"] == "present" and window.start <= c["frame"] <= window.end
        ),
        default=window.start,
    )
    reference = read_frame(result.video, reference_frame)
    direct_gray = False
    try:
        tracker = TextColorTracker(reference, result.rois["green"])
    except ValueError:
        if not window.confirmed:
            raise
        tracker = TextColorTracker.from_gray_reference(reference, result.rois["green"])
        if tracker is None:
            raise
        direct_gray = True
    del reference
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
        with closing(_scan_signals(result.video, window, extract, tracker, cancelled)) as scan:
            for index, (light, green_ratio, black_ratio), matches in scan:
                gm, bm = matches["green"], matches["black"]
                green, black = gm["score"], bm["score"]
                lights.append(light)
                greens.append(green)
                blacks.append(black)
                black_poses.append((*bm["position"], bm["scale"]))
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
    detection_fps = sampling_fps(result.video)
    args = (lights, greens, blacks, detection_fps, window.start, window.end + 1)
    detected = detect_rounds(*args, strict_pairs=True) if direct_gray else detect_rounds(*args)
    gray_tracker, recovered_poses = None, {}
    fixed_touch_starts = {}
    if direct_gray:
        detected.notes.append("原参考帧无法提取固定字形，已使用灰字模板并要求双灭灯配对。")
        detected.diagnostics["gray_recovery"] = {
            "initial_template": "gray",
            "reference": tracker.gray_reference,
            "extra_scan_frames": 0,
        }
    elif (
        window.confirmed
        and len(detected.rounds) < session.expected_rounds
        and len(detected.diagnostics["unmatched_flash_frames"]) >= 4
    ):
        # Retain the touch search windows of previously paired responses.
        previous = window.start
        for start, end in detected.diagnostics["green_segments"]:
            fixed_touch_starts[start] = previous
            previous = end
        gray_tracker, recovered_poses = recover_gray_rounds(
            detected,
            result.video,
            result.rois["green"],
            reference_frame,
            window,
            detection_fps,
            directory,
            progress,
            cancelled,
        )
    (directory / "detection.json").write_text(
        json.dumps(detected.diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result.rounds = detected.rounds
    result.analysis_notes.extend(detected.notes)
    percentage(45)
    gray_attempted = gray_tracker is not None or direct_gray
    if not result.rounds:
        if gray_attempted and session.rounds:
            retain_manual_fields(session.rounds, result.rounds, window.start)
        return result
    progress("准备原图触摸轮廓与读数…" if contact else "寻找每轮指腹，估计接近方向…")
    if result.press_mode == "auto":
        try:
            hand = create_hand_detector(image_mode=True)
        except (ImportError, FileNotFoundError, RuntimeError, ValueError) as error:
            logger.exception("手部模型初始化失败")
            result.analysis_notes.append(f"手部识别未启用：{error}")
    ocr_at = {}
    windows = []
    probes = {}
    earlier_greens = {}
    glyph = interior_glyph(tracker.glyph)
    gray_glyph = interior_glyph(gray_tracker.glyph) if recovered_poses else None
    manual_greens = {}
    for r, (item, end) in enumerate(zip(result.rounds, detected.green_ends)):
        blink = item.blink2.frame
        next_start = (
            result.rounds[r + 1].blink1.frame if r + 1 < len(result.rounds) else window.end + 1
        )
        # Match by original video time, including when the user also corrected a blink.
        previous = [
            old
            for old in session.rounds
            if old.green.state == "manual"
            and old.green.frame is not None
            and item.blink1.frame <= old.green.frame < next_start
        ]
        if len(previous) == 1 and not gray_attempted:
            manual_greens[r] = previous[0]
        if (
            blink is not None
            and blink >= window.start + 2
            and item.green.frame - blink > MIN_LEAD_FRAMES
            and r not in manual_greens
        ):
            if item.green.frame in recovered_poses:
                probes[r] = GreenOnsetProbe(
                    gray_glyph, recovered_poses[item.green.frame], blink, item.green.frame
                )
            else:
                anchor = max(
                    range(max(window.start, blink - 9), blink + 1),
                    key=lambda i: blacks[i] - greens[i],
                )
                probes[r] = GreenOnsetProbe(
                    glyph, black_poses[anchor - window.start], blink, item.green.frame
                )
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
            previous_end = fixed_touch_starts.get(item.green.frame, previous_end)
            stop = (
                min(end - 1, item.blink1.frame + max(3, round(detection_fps * 0.05)))
                if contact
                else min(end - 1, item.blink1.frame + max(3, round(detection_fps * 0.1)))
            )
            windows.append((previous_end, stop, r))
    motion_trackers, targets, motion_decisions, patches = {}, {}, {}, {}
    if not contact:
        try:
            with (
                (directory / "touch-targets.jsonl").open("w", encoding="utf-8") as target_log,
                closing(FrameReader(result.video)) as reader,
            ):
                for index, (start, stop, r) in enumerate(windows):
                    progress(f"寻找指腹样本：第{r + 1}/{len(result.rounds)}轮")
                    tracker_for_round, anchor, report = prepare_motion_tracker(
                        reader.read,
                        result.rois["touch"],
                        start,
                        stop,
                        result.rounds[r].blink1.frame,
                        detection_fps,
                        detector=hand,
                        sample=result.touch_calibration if result.press_mode == "sample" else None,
                        cancelled=cancelled,
                        progress=lambda frame, number=r: progress(
                            f"寻找指腹：第{number + 1}轮，原片第{frame}帧"
                        ),
                    )
                    targets[r] = report
                    target_log.write(
                        json.dumps({"round": r + 1, **report}, ensure_ascii=False) + "\n"
                    )
                    target_log.flush()
                    if tracker_for_round is not None:
                        motion_trackers[r] = (tracker_for_round, anchor)
                        patches[r] = []
                    else:
                        result.rounds[r].press = Event(reason=report["reason"])
                    percentage(45 + 5 * (index + 1) // len(windows))
        finally:
            if hand:
                hand.close()
    readings = [[] for _ in result.rounds]
    touch_samples = [[] for _ in result.rounds]
    touch_decisions = []
    # A rescued round can overlap an original round's retained touch window.
    separate_contacts = (
        {r: copy.copy(contact) for _, _, r in windows} if contact and recovered_poses else {}
    )
    probe_seconds = 0.0
    probe_frames = extra_display_frames = 0
    with (
        (directory / "ocr-checks.jsonl").open("w", encoding="utf-8") as reading_log,
        (directory / "touch-signals.csv").open("w", encoding="utf-8", newline="") as touch_log,
        closing(FrameReader(result.video)) as reader,
    ):
        touch_writer = csv.DictWriter(
            touch_log,
            fieldnames=[
                "round",
                "method",
                *TouchSample.__dataclass_fields__,
                "flow_points",
                "overlap",
                "reason",
                "flow_x",
                "flow_y",
                "position_px",
                "front_clipped",
            ],
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

        with av.open(result.video.path) as container:
            container.streams.video[0].thread_type = "AUTO"
            for i, frame in enumerate(container.decode(video=0)):
                if cancelled():
                    raise InterruptedError("已取消")
                if i > window.end:
                    break
                if i < window.start:
                    continue
                if i % 60 == 0:
                    percentage(50 + 25 * (i - window.start + 1) // (window.end - window.start + 1))
                    progress(
                        f"读取数字与分析触摸动作：{i - window.start + 1}/{window.end - window.start + 1}帧"
                    )
                active = [
                    r
                    for start, end, r in windows
                    if start <= i <= end and (contact or r in motion_trackers)
                ]
                probing = [
                    (r, probe) for r, probe in probes.items() if probe.start <= i <= probe.stop
                ]
                only_probes = not active and (not ocr or i not in ocr_at)
                if only_probes and not probing:
                    continue
                if only_probes:
                    extra_display_frames += 1
                    probe_started = perf_counter()
                image = display_image(frame)
                if probing:
                    if not only_probes:
                        probe_started = perf_counter()
                    search = crop(image, tracker.search_roi)
                    for r, probe in probing:
                        probe.observe(i, search)
                        if i == probe.stop:
                            if probe.candidate is not None:
                                earlier_greens[r] = probe.candidate
                            del probes[r]
                    probe_frames += 1
                    probe_seconds += perf_counter() - probe_started
                if ocr and i in ocr_at:
                    reading = read_numbers(ocr, crop(image, result.rois["reading"]))
                    for r in ocr_at[i]:
                        readings[r].append((i, *reading))
                        record_reading(r, i, "seed", reading)
                if active and contact:
                    for r in active:
                        finger = separate_contacts.get(r, contact)
                        if any(i == start and number == r for start, _, number in windows):
                            finger.previous_center = finger.reference_center.copy()
                        sample = finger.measure(i, image)
                        touch_samples[r].append(sample)
                        touch_writer.writerow(
                            {"round": r + 1, "method": "calibrated_contour", **asdict(sample)}
                        )
                elif active:
                    for r in active:
                        finger, anchor = motion_trackers[r]
                        patches[r].append((i, image[finger.bounds].copy()))
                        stop = next(end for _, end, number in windows if number == r)
                        if i == stop:
                            progress(f"判断指腹运动转折：第{r + 1}/{len(result.rounds)}轮")
                            observations = finger.trace(patches.pop(r), anchor, cancelled)
                            event, decision, rows = locate_motion_press(
                                observations,
                                detection_fps,
                                result.rounds[r].blink1.frame,
                                finger.origin,
                                finger.mask,
                            )
                            result.rounds[r].press = event
                            motion_decisions[r] = decision
                            for row in rows:
                                touch_writer.writerow(
                                    {"round": r + 1, "method": "automatic_finger_motion", **row}
                                )
                            ok, encoded = cv2.imencode(".png", motion_preview(finger, decision))
                            if ok:
                                (directory / f"touch-preview-{r + 1:02d}.png").write_bytes(
                                    encoded.tobytes()
                                )
                if active and i % 60 == 0:
                    touch_log.flush()
        percentage(75)
        neighbor_step = max(1, round(sampling_fps(result.video) / 24))
        reading_crops = {}

        def prefetch_readings(frames):
            reading_crops.clear()
            for frame in frames:
                if cancelled():
                    raise InterruptedError("已取消")
                try:
                    image = reader.read(frame)
                except (OSError, ValueError):
                    # A later crop might never be needed. Report a read failure
                    # only if OCR actually requests that frame.
                    break
                reading_crops[frame] = crop(image, result.rois["reading"])

        for r, item in enumerate(result.rounds):
            if cancelled():
                raise InterruptedError("已取消")
            progress(f"核对MD/TX：第{r + 1}/{len(result.rounds)}轮")

            def read_at(frame, round_index=r):
                patch = reading_crops.pop(frame, None)
                if patch is None:
                    patch = crop(reader.read(frame), result.rois["reading"])
                reading = read_numbers(ocr, patch)
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
                    prefetch=prefetch_readings if reader.seekable else None,
                )
            else:
                choose_reading(item, readings[r])
            if contact:
                item.press, decision = locate_contact(
                    touch_samples[r], detection_fps, item.blink1.frame
                )
            else:
                decision = {**motion_decisions.get(r, {}), "target": targets.get(r, {})}
            touch_decisions.append(
                {
                    "round": r + 1,
                    "mode": result.press_mode,
                    "blink1": item.blink1.frame,
                    "event": asdict(item.press),
                    **decision,
                }
            )
            percentage(75 + 20 * (r + 1) // len(result.rounds))
    (directory / "touch-detection.json").write_text(
        json.dumps(touch_decisions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Refine only after the original sampling passes: other fields keep their evidence.
    proofs = {row["first_green_candidate"]: row for row in detected.diagnostics["green_evidence"]}
    detected.diagnostics["coarse_green_segments"] = list(detected.diagnostics["green_segments"])
    refinements = []
    with (
        (directory / "green-refinement.jsonl").open("w", encoding="utf-8") as refinement_log,
        closing(FrameReader(result.video)) as reader,
    ):
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
                    gray_tracker if coarse in recovered_poses else tracker,
                    reader.read,
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
            saved = manual_greens.get(r)
            if saved is not None:
                item.green = copy.copy(saved.green)
                detected.diagnostics["round_candidates"][r]["retained_manual_green"] = (
                    item.green.frame
                )
                if (
                    saved.green_recommendation
                    and item.blink2.frame is not None
                    and item.blink2.frame < item.green.frame
                ):
                    item.green_recommendation = copy.deepcopy(saved.green_recommendation)
            elif r in earlier_greens:
                earlier = earlier_greens[r]
                if item.green.frame - earlier >= MIN_LEAD_FRAMES:
                    item.green_recommendation = GreenRecommendation(earlier, copy.copy(item.green))
                    detected.diagnostics["round_candidates"][r]["recommended_green"] = earlier
            record({"kind": "result", **refinement})
            percentage(95 + 4 * (r + 1) // len(result.rounds))
    detected.diagnostics["green_refinements"] = refinements
    detected.diagnostics["green_recommendation_pass"] = {
        "min_lead_frames": MIN_LEAD_FRAMES,
        "frames_checked": probe_frames,
        "extra_display_frames": extra_display_frames,
        "probe_seconds": round(probe_seconds, 3),
    }
    progress(
        f"green推荐检查：{probe_frames}帧，其中额外转图{extra_display_frames}帧，"
        f"笔画检查与额外转图共{probe_seconds:.2f}秒"
    )
    if gray_attempted and session.rounds:
        retain_manual_fields(session.rounds, result.rounds, window.start)
        for item, row in zip(result.rounds, detected.diagnostics["round_candidates"]):
            if item.green.state == "manual":
                row["retained_manual_green"] = item.green.frame
                row.pop("recommended_green", None)
    (directory / "detection.json").write_text(
        json.dumps(detected.diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    corrected = sum(row["applied"] for row in refinements)
    if corrected:
        result.analysis_notes.append(f"green笔画颜色回溯调整了{corrected}轮首帧，过渡边界保留复核")
    progress(f"分析完成：找到{len(result.rounds)}轮，疑难项可逐帧修正")
    return result
