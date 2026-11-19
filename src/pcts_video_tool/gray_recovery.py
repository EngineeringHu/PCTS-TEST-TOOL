"""Recover missing responses in gaps between the ordinary detector's paired rounds."""

import copy
import csv
from time import perf_counter

import av
import cv2
import numpy as np

from .detection import TextColorTracker, pair_responses
from .models import EVENTS
from .video import display_image, read_frame


class ManualRoundMismatch(ValueError):
    """The new evidence cannot safely identify an already confirmed measurement."""


def missing_windows(detected, start, stop, fps):
    """Protect each known flash/response interval, including unpaired green segments."""
    diag = detected.diagnostics
    paired = {r.green.frame: r.blink1.frame for r in detected.rounds}
    reset = max(12, round(fps * 0.1))
    windows = []
    lo = start
    for green, end in [*diag["green_segments"], (stop, stop)]:
        hi = paired.get(green, green)
        flashes = [i for i in diag["unmatched_flash_frames"] if lo <= i < hi]
        if len(flashes) >= 2:
            windows.append((max(lo, flashes[0] - reset), hi, flashes))
        lo = end
    return windows


def recover_gray_rounds(
    detected, video, roi, reference_frame, window, fps, directory, progress, cancelled
):
    """A single ordered replay measures only text in eligible gaps; flash edges stay fixed."""
    spans = missing_windows(detected, window.start, window.end + 1, fps)
    if not spans:
        return None, {}
    started = perf_counter()
    tracker = TextColorTracker.from_gray_reference(read_frame(video, reference_frame), roi)
    if tracker is None:
        return None, {}
    progress("检测到未配对灭灯，使用灰字模板检查缺失轮次…")
    scores = np.zeros((2, window.end + 1))
    poses = {}
    # Only the ten poses preceding a possible second flash can be needed by the green probe.
    pose_frames = {
        i for lo, _, flashes in spans for f in flashes for i in range(max(lo, f - 9), f + 1)
    }
    ok, picture = cv2.imencode(".png", tracker.glyph)
    if ok:
        (directory / "gray-template.png").write_bytes(picture.tobytes())
    with (
        av.open(video.path) as container,
        (directory / "gray-signals.csv").open("w", newline="", encoding="utf-8") as log,
    ):
        container.streams.video[0].thread_type = "AUTO"
        writer = csv.writer(log)
        writer.writerow(
            ["frame", "green_shape_score", "black_shape_score", "black_x", "black_y", "black_scale"]
        )
        span = 0
        for i, frame in enumerate(container.decode(video=0)):
            if cancelled():
                raise InterruptedError("已取消")
            if i >= spans[-1][1]:
                break
            if i >= spans[span][1]:
                span += 1
            if i < spans[span][0]:
                continue
            matches = tracker.match(display_image(frame))
            g, b = matches["green"]["score"], matches["black"]
            scores[:, i] = (g, b["score"])
            pose = (*b["position"], b["scale"])
            if i in pose_frames:
                poses[i] = pose
            writer.writerow([i, g, b["score"], *pose])

    diag = detected.diagnostics
    added_poses = {}
    consumed = set()
    original_count = len(detected.rounds)
    for lo, hi, flashes in spans:
        trial = pair_responses(flashes, *scores, fps, lo, hi, strict_pairs=True)
        evidence = {p["first_green_candidate"]: p for p in trial.diagnostics["green_evidence"]}
        for item, end, row in zip(
            trial.rounds, trial.green_ends, trial.diagnostics["round_candidates"]
        ):
            if hi <= window.end and end == hi:
                continue  # This response never reset before a protected interval.
            green, blink = item.green.frame, item.blink2.frame
            anchor = max(
                range(max(lo, blink - 9), blink + 1), key=lambda i: scores[1, i] - scores[0, i]
            )
            added_poses[green] = poses[anchor]
            consumed.update((item.blink1.frame, blink))
            detected.rounds.append(item)
            diag["green_segments"].append((green, end))
            diag["green_evidence"].append(evidence[green])
            diag["round_candidates"].append(dict(row, template="gray"))
    diag["gray_recovery"] = {
        "reference": tracker.gray_reference,
        "original_round_count": original_count,
        "added_green_frames": sorted(added_poses),
        "scan_ranges": [[lo, hi - 1] for lo, hi, _ in spans],
        "matched_frames": sum(hi - lo for lo, hi, _ in spans),
        "elapsed_seconds": round(perf_counter() - started, 3),
    }
    if added_poses:
        detected.rounds.sort(key=lambda r: r.green.frame)
        detected.green_ends = [r.last_green_frame + 1 for r in detected.rounds]
        diag["green_segments"].sort()
        diag["green_evidence"].sort(key=lambda p: p["first_green_candidate"])
        diag["round_candidates"].sort(key=lambda row: row["green"])
        diag["unmatched_flash_frames"] = [
            i for i in diag["unmatched_flash_frames"] if i not in consumed
        ]
        detected.notes = [
            f"原模板识别{original_count}轮，灰字模板补回{len(added_poses)}轮，共{len(detected.rounds)}轮。",
        ]
        if diag["unmatched_flash_frames"]:
            detected.notes.append(f"仍有未配对灭灯帧：{diag['unmatched_flash_frames']}")
        if diag["unmatched_green_frames"]:
            detected.notes.append(f"缺少灯光对应的变绿帧：{diag['unmatched_green_frames']}")
    else:
        detected.notes.append(
            "已检查灰字备用模板，未找到同时满足黑字复位、双灭灯与持续变绿的新轮次。"
        )
    return tracker, added_poses


def retain_manual_fields(previous, rounds, start):
    """Match confirmed fields by flash pair or original frame, never by the new table index."""
    used = set()
    for old in previous:
        events = [
            key
            for key in EVENTS
            if getattr(old, key).state == "manual" and getattr(old, key).frame is not None
        ]
        manual_reading = old.reading_state == "manual" or "manual" in old.reading_states.values()
        if not events and not manual_reading:
            continue
        matches = [
            i
            for i, r in enumerate(rounds)
            if old.blink1.frame is not None
            and old.blink2.frame is not None
            and (r.blink1.frame, r.blink2.frame) == (old.blink1.frame, old.blink2.frame)
        ]
        if not matches:
            anchor = next(
                (
                    f
                    for f in (
                        old.green.frame,
                        old.blink2.frame,
                        old.blink1.frame,
                        old.reading_frame,
                        old.press.frame,
                    )
                    if f is not None
                ),
                None,
            )
            if anchor is not None:
                matches = [
                    i
                    for i, r in enumerate(rounds)
                    if (rounds[i - 1].last_green_frame + 1 if i else start)
                    <= anchor
                    <= r.last_green_frame
                ]
        if len(matches) != 1 or matches[0] in used:
            raise ManualRoundMismatch("灰字恢复后无法唯一对应已有人工确认的轮次，已保留原任务。")
        index = matches[0]
        used.add(index)
        item = rounds[index]
        for name in events:
            setattr(item, name, copy.copy(getattr(old, name)))
        if "green" in events or "blink2" in events:
            # A changed manual anchor makes a newly computed suggestion stale.
            item.green_recommendation = None
            rec = old.green_recommendation
            if (
                rec
                and old.green.state == "manual"
                and item.blink2.frame is not None
                and item.blink2.frame < rec.frame == item.green.frame
            ):
                item.green_recommendation = copy.deepcopy(rec)
        if manual_reading:
            # The two numbers share one source frame and OCR evidence.
            item.md, item.tx, item.reading_frame = old.md, old.tx, old.reading_frame
            item.reading_state, item.reading_reason, item.ocr_text = (
                old.reading_state,
                old.reading_reason,
                old.ocr_text,
            )
            item.reading_states = old.reading_states.copy()
