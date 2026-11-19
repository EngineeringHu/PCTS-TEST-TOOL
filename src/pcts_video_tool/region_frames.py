"""Find original frames for manual ROI selection while importing a video."""

from __future__ import annotations

import re
from contextlib import closing

import cv2
import numpy as np

from .marker import marker_state
from .models import Session
from .motion import automatic_finger_samples
from .recognition import create_hand_detector, create_ocr
from .run_logs import logger
from .video import FrameReader, crop, inspect_video, sampling_fps


def _text_targets(result):
    marker = None
    reading = False
    lines = list(zip(result.txts or (), result.boxes if result.boxes is not None else ()))
    for i, (text, box) in enumerate(lines):
        if re.search(r"(?<![A-Z])(?:MD|TX)(?![A-Z])", text.upper()):
            reading = True
        if marker_state(text) == "present":
            marker = np.asarray(box, np.float32)
        elif text.strip().upper() == "PRESS" and i + 1 < len(lines):
            following, other = lines[i + 1]
            if following.strip().upper() == "HERE":
                marker = np.float32([box[0], other[1], other[2], box[3]])
    return marker, reading


def _skin_fraction(patch):
    mask = cv2.inRange(cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb), (0, 136, 70), (255, 183, 126))
    return np.mean(mask > 0)


def _sample_markers(thumbnails, reference, target, touch_area):
    template = cv2.cvtColor(crop(reference, target.ravel()), cv2.COLOR_BGR2GRAY)
    search = np.clip(target.mean(axis=0) + (target - target.mean(axis=0)) * 1.7, 0, 1)
    signals = []
    for _, picture in thumbnails:
        patch = crop(picture, target.ravel())
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, (35, 65, 35), (90, 255, 255))
        neutral = cv2.inRange(hsv, (0, 0, 110), (179, 85, 255))
        nearby = cv2.cvtColor(crop(picture, search.ravel()), cv2.COLOR_BGR2GRAY)
        match = cv2.minMaxLoc(cv2.matchTemplate(nearby, template, cv2.TM_CCOEFF_NORMED))[1]
        signals.append(
            (
                np.mean(green > 0),
                np.mean(neutral > 0),
                match,
                _skin_fraction(crop(picture, touch_area.ravel())),
            )
        )
    return np.array(signals).T


def _find_region_frames(thumbnails, reader, frames, progress, cancelled):
    if cancelled():
        raise InterruptedError("已取消")
    progress("正在准备框选画面：识别 PRESS HERE 和 MD / TX…")
    engine = create_ocr()
    count = len(thumbnails)
    probes = [round((count - 1) * f) for f in (1 / 3, 1 / 2, 2 / 3, 1 / 4, 3 / 4, 1 / 8, 7 / 8)]
    checked = set()
    target = None
    signals = None
    while probes and len(checked) < 8:
        if cancelled():
            raise InterruptedError("已取消")
        pos = probes.pop(0)
        if pos in checked:
            continue
        checked.add(pos)
        index, thumbnail = thumbnails[pos]
        image = reader.read(index)
        box, reading = _text_targets(engine(image))
        if reading:
            frames.setdefault("reading", index)
        if box is not None:
            h, w = image.shape[:2]
            quad = np.clip(box / [w - 1, h - 1], 0, 1)
            if _skin_fraction(crop(image, quad.ravel())) < 0.04:
                frames.setdefault("green", index)
            if target is None:
                target = quad
                row = (quad[2] + quad[3] - quad[0] - quad[1]) / 2
                # A side view can put the fingertip just below the letters.
                touch_area = np.clip(quad + np.array([-row * 0.6] * 2 + [row * 1.3] * 2), 0, 1)
                signals = _sample_markers(thumbnails, thumbnail, quad, touch_area)
                green, neutral, match, skin = signals
                # Look late in a legible green run, when the result has appeared.
                quality = green * neutral * match - 0.1 * skin
                next_probes = []
                for candidate in np.argsort(-quality):
                    if all(abs(candidate - prev) > 2 for prev in next_probes):
                        next_probes.append(int(candidate))
                    if len(next_probes) == 3:
                        break
                probes[:0] = next_probes
        if target is not None and "reading" not in frames and signals[0, pos] > 0.05:
            # Rectify the neighboring result line when full-frame OCR misses an oblique label.
            side = (target[1] - target[0]) * 0.45
            area = target + [-side - 2.4 * row, side - 2.4 * row, side, -side]
            patch = crop(image, np.clip(area, 0, 1).ravel())
            _, reading = _text_targets(engine(patch))
            if reading:
                frames["reading"] = index
        if "green" in frames and "reading" in frames:
            break

    if target is None:
        return
    progress("正在准备框选画面：查找指腹点击屏幕的候选帧…")
    green, neutral, match, skin = signals
    # Favor the actual lettering on a plain screen over colored animation or a camera pan.
    visible = match > 0.35
    green_on = (green > 0.06) & visible
    onsets = np.flatnonzero(green_on & ~np.r_[True, green_on[:-1]])
    rank = skin * neutral * np.maximum(match, 0)
    indices = np.array([index for index, _ in thumbnails])
    fps = sampling_fps(reader.info)
    for pos in onsets:
        if pos and visible[pos - 1]:
            # The screen responds after the tap. Favor fingers still on the glass
            # over the withdrawing hand visible on the first green sample.
            onset = (indices[pos - 1] + indices[pos]) / 2
            proximity = np.maximum(0, 1 - np.abs((onset - indices) / fps - 0.14) / 0.14)
            rank += proximity * neutral[pos] * match[pos]
    rank[~visible] = -1
    with create_hand_detector(image_mode=True) as detector:
        for pos in np.argsort(-rank, kind="stable")[:12]:
            if rank[pos] <= 0:
                break
            if cancelled():
                raise InterruptedError("已取消")
            index, _ = thumbnails[pos]
            image = reader.read(index)
            scale = min(1, 768 / max(image.shape[:2]))
            preview = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if automatic_finger_samples(detector, preview, touch_area.ravel()):
                frames["touch"] = index
                break


def prepare_region_session(path, progress, cancelled, session=None):
    """Prepare transient, original-resolution previews without editing the saved task."""
    thumbnails = []
    info = inspect_video(path, progress, cancelled, thumbnails=thumbnails)
    with closing(FrameReader(info)) as reader:
        frames = {"flash": 0}
        try:
            _find_region_frames(thumbnails, reader, frames, progress, cancelled)
        except InterruptedError:
            raise
        except Exception:
            # A failure in optional recognition must leave manual selection usable.
            logger.exception("辅助选帧未完成，保留已找到的候选帧")
        previews = {}
        for name, index in frames.items():
            if cancelled():
                raise InterruptedError("已取消")
            previews[name] = (index, reader.read(index))
    return session or Session(info), previews
