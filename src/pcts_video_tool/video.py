from __future__ import annotations

from bisect import bisect_left
from collections import OrderedDict
from pathlib import Path

import av
import cv2
import numpy as np

from .models import VideoInfo


def display_image(frame: av.VideoFrame, max_side: int | None = None) -> np.ndarray:
    rotation = int(frame.rotation)
    if rotation % 90:
        raise ValueError(f"暂不支持{rotation}度非直角旋转视频")
    if max_side and max(frame.width, frame.height) > max_side:
        scale = max_side / max(frame.width, frame.height)
        frame = frame.reformat(
            width=max(1, round(frame.width * scale)),
            height=max(1, round(frame.height * scale)),
            format="bgr24",
        )
    image = frame.to_ndarray(format="bgr24")
    return np.ascontiguousarray(np.rot90(image, rotation // 90))


def read_frame(info: VideoInfo, index: int) -> np.ndarray:
    """Exact presentation index; sequential fallback also works without unique PTS."""
    with av.open(info.path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        target = info.times[index]
        if target is not None and all(
            b is not None and a is not None and b > a for a, b in zip(info.times, info.times[1:])
        ):
            container.seek(int(target / stream.time_base), stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.pts is not None:
                    t = float(frame.pts * frame.time_base)
                    if abs(t - target) < 1e-8:
                        return display_image(frame)
                    if t > target + 1e-8:
                        break
    with av.open(info.path) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i == index:
                return display_image(frame)
    raise ValueError(f"无法读取原片第{index}帧")


class FrameReader:
    """Exact-frame reader with its own decoder and a bounded raw-image cache."""

    def __init__(self, info: VideoInfo, max_bytes: int = 96 * 1024 * 1024):
        self.info = info
        self.max_bytes = max_bytes
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.container = None
        self.stream = None
        self.decoder = None
        self.last_index = -1
        self.seekable = all(t is not None for t in info.times) and all(
            b > a for a, b in zip(info.times, info.times[1:])
        )

    def _remember(self, index, image):
        old = self.cache.pop(index, None)
        if old is not None:
            self.cache_bytes -= old.nbytes
        self.cache[index] = image
        self.cache_bytes += image.nbytes
        while self.cache_bytes > self.max_bytes:
            _, removed = self.cache.popitem(last=False)
            self.cache_bytes -= removed.nbytes
        return image

    def read(self, index: int) -> np.ndarray:
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        if not self.seekable:
            return self._remember(index, read_frame(self.info, index))
        if self.container is None:
            self.container = av.open(self.info.path)
            self.stream = self.container.streams.video[0]
            self.stream.thread_type = "AUTO"
            self.decoder = iter(self.container.decode(self.stream))
            self.last_index = -1
        if not self.last_index < index <= self.last_index + 32:
            target = self.info.times[index]
            self.container.seek(
                int(target / self.stream.time_base), stream=self.stream, backward=True
            )
            # seek flushes the codec, but a suspended generator may still hold old decoded frames.
            self.decoder = iter(self.container.decode(self.stream))
        for frame in self.decoder:
            if frame.pts is None:
                break
            timestamp = float(frame.pts * frame.time_base)
            position = bisect_left(self.info.times, timestamp - 1e-8)
            if (
                position >= len(self.info.times)
                or abs(self.info.times[position] - timestamp) >= 1e-8
            ):
                break
            self.last_index = position
            if position > index:
                break
            if position >= index - 8:
                image = self._remember(position, display_image(frame))
                if position == index:
                    return image
        # Preserve the established presentation-index fallback if a seek cannot be aligned.
        return self._remember(index, read_frame(self.info, index))

    def close(self):
        self.decoder = None
        if self.container is not None:
            self.container.close()
        self.container = self.stream = None
        self.cache.clear()
        self.cache_bytes = 0
        self.last_index = -1


def inspect_video(
    path: str, progress=None, cancelled=None, sample=None, *, thumbnails=None
) -> VideoInfo:
    times = []
    source = Path(path).resolve()
    with av.open(str(source)) as container:
        if not container.streams.video:
            raise ValueError("文件没有视频轨道")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        average = float(stream.average_rate or 0)
        nominal = float(stream.base_rate or 0)
        width, height, rotation = stream.width, stream.height, 0
        stride = max(1, (stream.frames + 191) // 192)
        for index, frame in enumerate(container.decode(stream)):
            if cancelled and cancelled():
                raise InterruptedError("已取消")
            times.append(None if frame.pts is None else float(frame.pts * frame.time_base))
            if index == 0 or sample:
                image = display_image(frame)
                height, width = image.shape[:2]
                rotation = int(frame.rotation)
                if sample:
                    sample(index, image)
            if thumbnails is not None and index % stride == 0:
                # Some containers omit the frame count. Keep sampling the entire timeline.
                if len(thumbnails) == 192:
                    del thumbnails[1::2]
                    stride *= 2
                if index % stride == 0:
                    thumbnails.append((index, display_image(frame, max_side=384)))
            if progress and index % 120 == 0:
                progress(f"正在逐帧读取：{index + 1} / {stream.frames or '?'}")
    if not times:
        raise ValueError("视频中没有可解码画面")
    if len(times) > 1 and times[0] is not None and times[-1] is not None and times[-1] > times[0]:
        average = (len(times) - 1) / (times[-1] - times[0])
    stat = source.stat()
    info = VideoInfo(
        str(source),
        width,
        height,
        stat.st_size,
        stat.st_mtime_ns,
        times,
        nominal,
        average,
        rotation,
    )
    intervals = [b - a for a, b in zip(times, times[1:]) if a is not None and b is not None]
    if any(t is None for t in times):
        info.warnings.append("部分帧缺少时间戳；涉及这些帧的时间戳计算不可用")
    if any(d <= 0 for d in intervals):
        info.warnings.append("时间戳不严格递增，请核实文件时间轴")
    long_count = sum(d > 1.5 / 240 for d in intervals)
    if long_count:
        info.warnings.append(f"相对240 fps存在{long_count}处长间隔（>6.25 ms）")
    return info


def rate_summary(info: VideoInfo, fps: float) -> str:
    deltas = np.array(
        [b - a for a, b in zip(info.times, info.times[1:]) if a is not None and b is not None]
    )
    head = (
        f"{info.width}×{info.height} · {len(info.times)}帧 · "
        f"帧间隔估算{sampling_fps(info):.3f} fps · 时间轴平均{info.average_fps:.3f} fps"
    )
    if not len(deltas):
        return head + "\n没有足够时间戳可检查帧间隔"
    long_count = int(np.count_nonzero(deltas > 1.5 / fps))
    return (
        head + f"\n相邻帧间隔 {deltas.min() * 1000:.3f}～{deltas.max() * 1000:.3f} ms；"
        f"相对{fps:g} fps的长间隔 {long_count}处。帧号从0开始。"
    )


def sampling_fps(info: VideoInfo) -> float:
    """Detection cadence comes from frame timestamps, never FFmpeg's base-rate guess."""
    deltas = [
        b - a
        for a, b in zip(info.times, info.times[1:])
        if a is not None and b is not None and b > a
    ]
    return 1 / float(np.median(deltas)) if deltas else (info.average_fps or 240.0)


def crop(image: np.ndarray, roi: list[float]) -> np.ndarray:
    """Rectify a four-corner region for OCR; frame numbering is unchanged."""
    h, w = image.shape[:2]
    points = np.array(roi, np.float32).reshape(4, 2) * [w - 1, h - 1]
    out_w = max(
        2,
        round(max(np.linalg.norm(points[1] - points[0]), np.linalg.norm(points[2] - points[3])))
        + 1,
    )
    out_h = max(
        2,
        round(max(np.linalg.norm(points[3] - points[0]), np.linalg.norm(points[2] - points[1])))
        + 1,
    )
    target = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    matrix = cv2.getPerspectiveTransform(points.astype(np.float32), target)
    return cv2.warpPerspective(image, matrix, (out_w, out_h), borderMode=cv2.BORDER_REPLICATE)
