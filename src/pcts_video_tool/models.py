from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

EVENTS = ("press", "blink1", "blink2", "green")
METRICS = {
    "end_to_end": "端到端",
    "hu_input": "HU input",
    "blink_interval": "两次灭灯间隔",
    "hu_output": "HU output",
    "md_total": "MD total",
    "hu_total": "HU total",
    "flash_accuracy": "闪光时间差",
}


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    size: int
    mtime_ns: int
    times: list[float | None]
    nominal_fps: float = 0.0
    average_fps: float = 0.0
    rotation: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class Event:
    frame: int | None = None
    state: str = "review"
    reason: str = "尚未识别"


@dataclass
class TouchCalibration:
    frame: int
    line: list[float]
    skin_point: list[float]


@dataclass
class Round:
    press: Event = field(default_factory=Event)
    blink1: Event = field(default_factory=Event)
    blink2: Event = field(default_factory=Event)
    green: Event = field(default_factory=Event)
    md: float | None = None
    tx: float | None = None
    reading_frame: int | None = None
    reading_state: str = "review"
    reading_reason: str = "尚未识别"
    ocr_text: str = ""
    issues: list[str] = field(default_factory=list)

    def problems(self) -> list[str]:
        problems = list(self.issues)
        for name in EVENTS:
            event = getattr(self, name)
            if event.frame is None or event.state == "review":
                problems.append(f"{name}: {event.reason or '待确认'}")
        frames = [getattr(self, name).frame for name in EVENTS]
        if all(v is not None for v in frames) and frames != sorted(frames):
            problems.append("事件顺序应为 press ≤ blink1 ≤ blink2 ≤ green")
        if self.md is None or self.tx is None or self.reading_state == "review":
            problems.append(f"MD/TX: {self.reading_reason or '待确认'}")
        if self.md is not None and self.tx is not None and self.reading_frame is None:
            problems.append("MD/TX读数来源帧未指定")
        return problems


@dataclass
class Session:
    video: VideoInfo
    rois: dict[str, list[float]] = field(default_factory=dict)
    rounds: list[Round] = field(default_factory=list)
    method: str = "frames"
    fps: float = 240.0
    expected_rounds: int = 10
    analysis_notes: list[str] = field(default_factory=list)
    diagnostic_dir: str | None = None
    range_mode: str = "auto"
    test_start: int | None = None
    test_end: int | None = None
    range_confirmed: bool = False
    touch_calibration: TouchCalibration | None = None

    def problems(self) -> list[str]:
        result = []
        if not self.range_confirmed:
            result.append("有效测试区间待确认：可自动识别PRESS HERE或手工设置")
        if len(self.rounds) != self.expected_rounds:
            result.append(f"预期{self.expected_rounds}轮，当前{len(self.rounds)}轮")
        for i, item in enumerate(self.rounds, 1):
            result.extend(f"第{i}轮 {p}" for p in item.problems())
            if self.test_start is not None and self.test_end is not None:
                chosen = [getattr(item, k).frame for k in EVENTS] + [item.reading_frame]
                if any(f is not None and not self.test_start <= f <= self.test_end for f in chosen):
                    result.append(f"第{i}轮有字段超出有效测试区间")
            if self.method == "timestamps":
                chosen = [getattr(item, k).frame for k in EVENTS]
                if any(f is not None and self.video.times[f] is None for f in chosen):
                    result.append(f"第{i}轮所选帧缺少时间戳")
        times = self.video.times[
            self.test_start : (self.test_end + 1 if self.test_end is not None else None)
        ]
        if self.method == "timestamps" and any(
            b is not None and a is not None and b <= a for a, b in zip(times, times[1:])
        ):
            result.append("视频时间戳不严格递增，时间戳计算待核实")
        return result


def calculate(session: Session, item: Round) -> dict[str, float | None]:
    def elapsed(first: Event, last: Event) -> float | None:
        if first.frame is None or last.frame is None:
            return None
        if session.method == "frames":
            return (last.frame - first.frame) * 1000.0 / session.fps
        a, b = session.video.times[first.frame], session.video.times[last.frame]
        return None if a is None or b is None else (b - a) * 1000.0

    total = elapsed(item.press, item.green)
    blink = elapsed(item.blink1, item.blink2)
    md_total = None if item.md is None or item.tx is None else item.md + item.tx
    return {
        "end_to_end": total,
        "hu_input": elapsed(item.press, item.blink1),
        "blink_interval": blink,
        "hu_output": elapsed(item.blink2, item.green),
        "md_total": md_total,
        "hu_total": None if total is None or md_total is None else total - md_total,
        "flash_accuracy": None if md_total is None or blink is None else md_total - blink,
    }


def save_session(session: Session, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"format": 4, **asdict(session)}, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_session(path: Path) -> Session:
    """Validate external session data once at the file boundary."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.pop("format", None) not in (1, 2, 3, 4):
        raise ValueError("不支持的任务文件版本")
    video = VideoInfo(**data.pop("video"))
    if not video.times or video.width <= 0 or video.height <= 0:
        raise ValueError("任务缺少有效视频索引")
    if any(
        t is not None and (not isinstance(t, (int, float)) or not math.isfinite(t))
        for t in video.times
    ):
        raise ValueError("时间戳必须为有限数字或空值")
    rounds = []
    for raw in data.pop("rounds"):
        events = {}
        for name in EVENTS:
            event = Event(**raw.pop(name))
            if event.frame is not None and (
                type(event.frame) is not int or not 0 <= event.frame < len(video.times)
            ):
                raise ValueError(f"{name}帧号超出原片范围")
            if event.state not in ("auto", "manual", "review"):
                raise ValueError("未知事件状态")
            events[name] = event
        item = Round(**events, **raw)
        for value in (item.md, item.tx):
            if value is not None and (
                not isinstance(value, (int, float)) or not math.isfinite(value)
            ):
                raise ValueError("MD/TX必须为有限数字或空值")
        if item.reading_frame is not None and (
            type(item.reading_frame) is not int or not 0 <= item.reading_frame < len(video.times)
        ):
            raise ValueError("读数来源帧超出原片范围")
        if item.reading_state not in ("auto", "manual", "review"):
            raise ValueError("未知读数状态")
        rounds.append(item)
    raw_calibration = data.pop("touch_calibration", None)
    calibration = TouchCalibration(**raw_calibration) if raw_calibration is not None else None
    session = Session(video=video, rounds=rounds, touch_calibration=calibration, **data)
    if session.method not in ("frames", "timestamps"):
        raise ValueError("未知计算方式")
    if not math.isfinite(session.fps) or session.fps <= 0:
        raise ValueError("FPS必须大于0")
    if session.expected_rounds != 10:
        raise ValueError("本版本预期轮数固定为10")
    if session.range_mode not in ("auto", "manual") or type(session.range_confirmed) is not bool:
        raise ValueError("未知测试区间模式")
    bounds = (session.test_start, session.test_end)
    if any(v is not None for v in bounds):
        if not all(type(v) is int for v in bounds) or not 0 <= bounds[0] <= bounds[1] < len(
            video.times
        ):
            raise ValueError("有效测试区间超出原片范围")
    elif session.range_mode == "manual" or session.range_confirmed:
        raise ValueError("已确认的测试区间必须有起止帧")
    for name, roi in session.rois.items():
        if name not in ("flash", "green", "reading", "touch"):
            raise ValueError("未知区域配置")
        session.rois[name] = normalize_roi(roi, video.width, video.height)
    if calibration is not None:
        validate_touch_calibration(calibration, video, session.rois.get("touch"))
    return session


def validate_touch_calibration(
    calibration: TouchCalibration, video: VideoInfo, roi: list[float] | None
) -> None:
    """UI/file input boundary; measurement code receives validated coordinates."""
    if roi is None:
        raise ValueError("触摸校准需要先框选触摸观察区域")
    if type(calibration.frame) is not int or not 0 <= calibration.frame < len(video.times):
        raise ValueError("触摸校准帧超出原片范围")
    if (
        len(calibration.line) != 4
        or len(calibration.skin_point) != 2
        or not all(
            isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1
            for v in calibration.line + calibration.skin_point
        )
    ):
        raise ValueError("触摸校准需要画面内的参考线两端和指腹取样点")
    x1, y1, x2, y2 = calibration.line
    dx, dy = (x2 - x1) * (video.width - 1), (y2 - y1) * (video.height - 1)
    length = math.hypot(dx, dy)
    if length < 6:
        raise ValueError("接触参考线太短，请在原图选择至少6像素的线段")
    px, py = calibration.skin_point
    distance = (
        abs(dx * (py - y1) * (video.height - 1) - dy * (px - x1) * (video.width - 1)) / length
    )
    if distance < 2:
        raise ValueError("取样点应在指腹内部，不能位于参考线上")
    points = list(zip(roi[::2], roi[1::2]))
    sides = [
        (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])
        for a, b in zip(points, points[1:] + points[:1])
    ]
    if not (all(v >= 0 for v in sides) or all(v <= 0 for v in sides)):
        raise ValueError("指腹取样点必须位于触摸观察区域内")


def normalize_roi(values: list[float], width: int, height: int) -> list[float]:
    """ROI input boundary: migrate rectangles, validate a convex four-corner polygon."""
    if len(values) not in (4, 8) or not all(
        isinstance(v, (int, float)) and math.isfinite(v) for v in values
    ):
        raise ValueError("区域需要4个有限数值的角点")
    if len(values) == 4:
        x, y, w, h = values
        if w <= 0 or h <= 0:
            raise ValueError("区域宽高必须为正数")
        values = [x, y, x + w, y, x + w, y + h, x, y + h]
    if not all(0 <= v <= 1 for v in values):
        raise ValueError("区域超出画面")
    points = list(zip(values[::2], values[1::2]))
    cross = []
    for i in range(4):
        a, b, c = points[i], points[(i + 1) % 4], points[(i + 2) % 4]
        cross.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    if not (all(v > 1e-10 for v in cross) or all(v < -1e-10 for v in cross)):
        raise ValueError("四点不能交叉或重合，请依次选择左上、右上、右下、左下")
    area = (
        abs(
            sum(
                points[i][0] * points[(i + 1) % 4][1] - points[(i + 1) % 4][0] * points[i][1]
                for i in range(4)
            )
        )
        / 2
    )
    if area * width * height < 16:
        raise ValueError("选区太小，请重新框选")
    if cross[0] < 0:
        points = [points[0], points[3], points[2], points[1]]
    return [float(v) for point in points for v in point]


def check_source(info: VideoInfo) -> None:
    stat = Path(info.path).stat()
    if stat.st_size != info.size or stat.st_mtime_ns != info.mtime_ns:
        raise ValueError("原视频大小或修改时间已变化，请重新导入，不能复用旧帧号")
