from __future__ import annotations

import re
from dataclasses import dataclass

import av

from .models import VideoInfo
from .recognition import read_text
from .video import crop, display_image, read_frame, sampling_fps


@dataclass
class TestWindow:
    start: int | None
    end: int | None
    confirmed: bool
    reason: str
    checks: list[dict]


def marker_state(text: str) -> str:
    letters = re.sub(r"[^A-Z]", "", text.upper())
    if "PRESSHERE" in letters:
        return "present"
    if "PRESS" in letters or "HERE" in letters:
        return "partial"
    return "absent"


def find_test_window(
    info: VideoInfo,
    roi: list[float],
    engine,
    progress=lambda message: None,
    cancelled=lambda: False,
    record_check=lambda value: None,
    percentage=lambda value: None,
) -> TestWindow:
    """Find the first marker lifecycle; short/partial occlusions do not end it."""
    step = max(1, round(sampling_fps(info) / 4))
    checks, cache = [], {}

    def check(index, image=None, phase="scan"):
        if cancelled():
            raise InterruptedError("已取消")
        if index not in cache:
            image = read_frame(info, index) if image is None else image
            text, score = read_text(engine, crop(image, roi))
            state = marker_state(text)
            cache[index] = state
            checks.append(
                {
                    "frame": index,
                    "timestamp_s": info.times[index],
                    "text": text,
                    "score": score,
                    "state": state,
                    "phase": phase,
                }
            )
            record_check(checks[-1])
        return cache[index]

    def boundary(low, high, entering):
        # Refine the interval between sampled states using original frame indices.
        while high - low > 1:
            mid = (low + high) // 2
            visible = check(mid, phase="refine") != "absent"
            if visible == entering:
                high = mid
            else:
                low = mid
        return high

    first, last_absent, last_visible = None, -1, None
    missing = []
    with av.open(info.path) as container:
        container.streams.video[0].thread_type = "AUTO"
        for index, frame in enumerate(container.decode(video=0)):
            if cancelled():
                raise InterruptedError("已取消")
            if index % step and index != len(info.times) - 1:
                continue
            progress(f"检查PRESS HERE起止：{index + 1}/{len(info.times)}帧")
            percentage((index + 1) * 100 // len(info.times))
            state = check(index, display_image(frame))
            if first is None:
                if state == "present":
                    first = boundary(last_absent, index, True) if index else 0
                    last_visible = index
                elif state == "absent":
                    last_absent = index
            elif state != "absent":
                missing = []
                last_visible = index
            else:
                missing.append(index)
                if len(missing) >= 4:
                    vanished = boundary(last_visible, missing[0], False)
                    return TestWindow(
                        first,
                        vanished - 1,
                        True,
                        f"PRESS HERE在第{first}帧出现、第{vanished}帧消失；连续4次检查确认消失",
                        checks,
                    )
    if first is None:
        return TestWindow(
            None, None, False, "未识别到完整PRESS HERE；请检查文字选区，或使用手工有效片段", checks
        )
    if missing:
        return TestWindow(
            first,
            len(info.times) - 1,
            False,
            "视频末尾文字无法识别，但持续时间不足以确认结束；请复核或手工设置有效片段",
            checks,
        )
    return TestWindow(
        first,
        len(info.times) - 1,
        True,
        f"PRESS HERE从第{first}帧出现，文件结束时仍可见，使用原片末帧作为可用终点",
        checks,
    )
