from copy import deepcopy
from fractions import Fraction

import av
import cv2
import numpy as np
import pytest

from pcts_video_tool import video
from pcts_video_tool.video import FrameReader, inspect_video, read_frame


@pytest.fixture
def indexed_video(tmp_path):
    path = tmp_path / "unique-frames.mp4"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=240)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        stream.gop_size = 12
        stream.codec_context.max_b_frames = 2
        for i in range(96):
            picture = np.full((64, 96, 3), (i * 2, 180 - i, 40 + i), np.uint8)
            cv2.putText(picture, str(i), (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            frame = av.VideoFrame.from_ndarray(picture, format="bgr24")
            frame.pts = i * 10 + (20 if i >= 31 else 0)
            frame.time_base = Fraction(1, 2400)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return inspect_video(str(path))


def test_seek_and_eviction_preserve_exact_pixels_with_b_frames_and_vfr(indexed_video):
    info = indexed_video
    assert info.times[31] - info.times[30] > info.times[30] - info.times[29]
    size = info.width * info.height * 3
    reader = FrameReader(info, max_bytes=size * 3)
    order = [0, 1, 2, 60, 61, 59, 58, 90, 91, 24, 25, 26, 0, 95, 31, 30, 32]
    for frame in order:
        assert np.array_equal(reader.read(frame), read_frame(info, frame)), frame
        assert reader.cache_bytes == sum(image.nbytes for image in reader.cache.values())
        assert reader.cache_bytes <= size * 3
    reader.close()
    assert reader.container is None and reader.decoder is None and not reader.cache
    assert reader.cache_bytes == 0


@pytest.mark.parametrize("kind", ["missing", "duplicate", "backwards"])
def test_ambiguous_timestamps_use_presentation_order_and_never_deduplicate(indexed_video, kind):
    info = deepcopy(indexed_video)
    if kind == "missing":
        info.times[8] = None
    elif kind == "duplicate":
        info.times[8] = info.times[7]
    else:
        info.times[8] = info.times[7] - 0.1
    reader = FrameReader(info, max_bytes=info.width * info.height * 3 * 3)
    for frame in (8, 7, 9, 60, 8, 7, 95, 0):
        assert np.array_equal(reader.read(frame), read_frame(indexed_video, frame)), frame
    reader.close()


def test_reuses_one_container_and_cached_frames_without_decoding_again(indexed_video, monkeypatch):
    original_open = video.av.open
    calls = []

    def opened(*args, **kwargs):
        calls.append(args[0])
        return original_open(*args, **kwargs)

    monkeypatch.setattr(video.av, "open", opened)
    reader = FrameReader(indexed_video)
    for frame in (50, 51, 52, 53, 54, 51, 52, 20, 21, 22):
        reader.read(frame)
    assert len(calls) == 1
    image = reader.read(21)
    assert reader.read(21) is image and len(calls) == 1
    reader.close()
    reader.read(21)
    assert len(calls) == 2
    reader.close()


def test_failed_timestamp_alignment_keeps_the_existing_exact_index_fallback(indexed_video):
    info = deepcopy(indexed_video)
    info.times = [t + 0.001 for t in info.times]
    reader = FrameReader(info)
    for index in (45, 2, 46):
        assert np.array_equal(reader.read(index), read_frame(indexed_video, index))
    reader.close()


def test_changing_video_does_not_reuse_the_same_frame_number(indexed_video, generated_video):
    from PySide6.QtWidgets import QApplication

    from pcts_video_tool.models import Session
    from pcts_video_tool.ui import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.accept_session(Session(indexed_video))
    old = window.frame_reader
    first = old.read(0)
    different = inspect_video(str(generated_video[0]))
    window.accept_session(Session(different))
    assert not old.cache and old.container is None
    current = window.frame_reader.read(0)
    assert current.shape != first.shape
    assert np.array_equal(current, read_frame(different, 0))
    assert window.canvas.pixmap.width() == different.width
    window.dirty = False
    window.close()
    app.processEvents()
