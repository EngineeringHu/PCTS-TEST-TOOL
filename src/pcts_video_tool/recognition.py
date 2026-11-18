from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .models import Event, Round

MAX_READING_RETRIES = 48


def model_directory() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "models"
    return Path(__file__).resolve().parents[2] / "models"


def create_ocr():
    import rapidocr
    from rapidocr import RapidOCR

    directory = Path(rapidocr.__file__).resolve().parent / "models"
    files = {
        "Det": "PP-OCRv6_det_small.onnx",
        "Rec": "PP-OCRv6_rec_small.onnx",
        "Cls": "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    }
    params = {
        "Global.use_cls": False,
        "Det.limit_type": "max",
        "Det.limit_side_len": 960,
        "EngineConfig.onnxruntime.use_cuda": False,
        "EngineConfig.onnxruntime.use_dml": False,
        "EngineConfig.onnxruntime.intra_op_num_threads": 4,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
    }
    for name, filename in files.items():
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"缺少本地OCR模型：{path}；请重新安装依赖，运行时不会下载")
        params[f"{name}.model_path"] = str(path)
    return RapidOCR(params=params)


def create_hand_detector():
    import mediapipe as mp

    path = model_directory() / "hand_landmarker.task"
    if not path.is_file():
        raise FileNotFoundError("缺少手部模型，请先运行 scripts/prepare_models.py")
    return mp.tasks.vision.HandLandmarker.create_from_options(
        mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(path), delegate=mp.tasks.BaseOptions.Delegate.CPU
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )


def parse_reading(text: str) -> tuple[float, float] | None:
    text = text.upper().replace("−", "-").replace("：", ":")
    # Require both labels and whole numeric tokens; never truncate a bad decimal.
    number = r"([+-]?\d+(?:\.\d)?)"
    matches = re.findall(r"\bMD\s*:\s*" + number + r"\s*TX\s*:\s*" + number + r"(?=$|[\s;|])", text)
    if len(matches) != 1:
        return None
    return float(matches[0][0]), float(matches[0][1])


def read_text(engine, image: np.ndarray) -> tuple[str, float]:
    # Upscale small crops; OCR receives original pixels, never annotations.
    if image.shape[0] < 96:
        image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    result = engine(image)
    text = " ".join(result.txts or ())
    score = min(result.scores) if result.scores else 0.0
    return text, float(score)


def read_numbers(engine, image: np.ndarray) -> tuple[str, tuple[float, float] | None, float]:
    if image.shape[0] < 96:
        image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    result = engine(image)
    lines = list(zip(result.txts or (), result.scores or ()))
    numeric_lines = [
        (text, score) for text, score in lines if re.search(r"\b(?:MD|TX)\s*[:：]", text.upper())
    ]
    selected = numeric_lines or lines
    text = " ".join(text for text, _ in lines)
    value = parse_reading(" ".join(text for text, _ in selected))
    if value is None and len(selected) != len(lines):
        # A detector can split a label and its digits into separate boxes.
        selected = lines
        value = parse_reading(text)
    return text, value, float(min((score for _, score in selected), default=0.0))


def choose_reading(item: Round, readings: list[tuple[int, str, tuple | None, float]]) -> None:
    item.ocr_text = "\n".join(f"帧{f}: {text or '(未识别)'}" for f, text, _, _ in readings)
    valid = list({row[0]: row for row in readings if row[2] is not None}.values())
    if not valid:
        item.reading_reason = "未读到完整MD:数值 TX:数值（整数或一位小数），请复核"
        item.reading_frame = readings[-1][0] if readings else None
        return
    counts = Counter(value for _, _, value, _ in valid)
    value = max(counts, key=lambda v: (counts[v], max(row[3] for row in valid if row[2] == v)))
    chosen = max((row for row in valid if row[2] == value), key=lambda row: row[3])
    item.md, item.tx = value
    item.reading_frame = chosen[0]
    agreement = sum(row[2] == value and row[3] >= 0.85 for row in valid)
    if len(counts) == 1 and agreement >= 3:
        item.reading_state, item.reading_reason = "auto", f"同轮{agreement}个独立原帧读数一致"
    elif len(counts) > 1:
        candidates = "；".join(
            f"MD:{md:g} TX:{tx:g}（{count}帧）" for (md, tx), count in counts.most_common(4)
        )
        item.reading_state = "review"
        item.reading_reason = "读数冲突，请查看原帧确认。候选：" + candidates
    else:
        item.reading_state, item.reading_reason = "review", "读数冲突、有效帧不足或字符评分较低"


def supplement_reading(
    item: Round,
    readings: list[tuple[int, str, tuple | None, float]],
    start: int,
    stop: int,
    read_at,
    cancelled=lambda: False,
    progress=lambda frame, count: None,
    frame_step: int = 10,
) -> None:
    """Check both sides of failed seed frames, strictly inside this round."""
    choose_reading(item, readings)
    if item.reading_state == "auto":
        return
    seeds = sorted(readings, key=lambda row: (row[2] is not None and row[3] >= 0.85, row[0]))
    seen = {row[0] for row in readings}
    count = 0
    batches = [
        [seed - distance, seed + distance] for distance in (1, 2, 4, 8, 16) for seed, *_ in seeds
    ]
    # If nearby frames share the same obstruction, cover other times in this round.
    stride = max(frame_step, (stop - start + 17) // 18)
    batches.extend([frame] for frame in range(start + stride, stop, stride))
    for batch in batches:
        for frame in batch:
            if not start <= frame < stop or frame in seen:
                continue
            if cancelled():
                raise InterruptedError("已取消")
            seen.add(frame)
            count += 1
            progress(frame, count)
            readings.append((frame, *read_at(frame)))
        choose_reading(item, readings)
        if item.reading_state == "auto" or count >= MAX_READING_RETRIES:
            return


def touch_candidate(points: list[tuple[int, float, float]]) -> Event:
    """Propose motion boundaries; landmark position does not prove physical contact."""
    if len(points) < 7:
        return Event(reason="触摸区域中未持续检测到指尖，请逐帧确认")
    frames = np.array([p[0] for p in points])
    xy = np.array([p[1:] for p in points])
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.diff(frames)
    candidates = []
    for i in range(3, len(speed) - 2):
        if frames[i + 3] - frames[i - 3] > 8:
            continue
        before, after = float(np.median(speed[i - 3 : i])), float(np.median(speed[i : i + 3]))
        if before > 0.001 and after < before * 0.45:
            candidates.append((before - after, int(frames[i])))
    if not candidates:
        return Event(reason="没有明确的指尖运动转折，请逐帧选择接触首帧")
    _, frame = max(candidates)
    return Event(frame, "review", "指尖减速候选；关键点不能证明接触，需要确认")


def fingertip(detector, image: np.ndarray, timestamp_ms: int, roi: list[float]):
    import mediapipe as mp

    scaled = cv2.resize(image, (640, max(1, round(image.shape[0] * 640 / image.shape[1]))))
    rgb = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
    result = detector.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), timestamp_ms
    )
    polygon = np.asarray(roi, np.float32).reshape(4, 2)
    candidates = [
        hand[8]
        for hand in result.hand_landmarks
        if cv2.pointPolygonTest(polygon, (hand[8].x, hand[8].y), False) >= 0
    ]
    if len(candidates) != 1:
        return None
    return candidates[0].x, candidates[0].y
