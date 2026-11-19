from fractions import Fraction

import av
import numpy as np
import pytest

from pcts_video_tool import marker
from pcts_video_tool.detection import detect_rounds
from pcts_video_tool.video import inspect_video


@pytest.fixture
def indexed_video(tmp_path, request):
    path = tmp_path / "marker-timeline.nut"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=240)
        stream.width, stream.height, stream.pix_fmt = 64, 32, "bgr0"
        for i in range(900):
            image = np.full((32, 64, 3), (i % 256, i // 256, 50), np.uint8)
            if 120 <= i < getattr(request, "param", 0):
                image[8:24, 20:32] = (0, 210, 0)
                image[8:24, 32:44] = (160, 30, 80)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts, frame.time_base = i, Fraction(1, 240)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return inspect_video(str(path))


def test_marker_window_ignores_short_occlusion_and_later_restart(indexed_video, monkeypatch):
    indexed_video.nominal_fps = 90000  # FFmpeg's base-rate guess is not the frame cadence.

    def recognize(engine, image):
        index = int(image[0, 0, 0]) + int(image[0, 0, 1]) * 256
        if 20 <= index < 280 or 600 <= index < 800:
            return ("" if 120 <= index < 170 else "PRESS HERE"), 0.99
        return "HOME", 0.99

    monkeypatch.setattr(marker, "read_text", recognize)
    window = marker.find_test_window(indexed_video, [0, 0, 1, 0, 1, 1, 0, 1], object())
    assert (window.start, window.end, window.confirmed) == (20, 279, True)
    assert max(c["frame"] for c in window.checks) < 600


def test_partial_text_does_not_end_test(indexed_video, monkeypatch):
    def recognize(engine, image):
        index = int(image[0, 0, 0]) + int(image[0, 0, 1]) * 256
        return ("PRESS HERE" if index < 120 else "HERE"), 0.95

    monkeypatch.setattr(marker, "read_text", recognize)
    window = marker.find_test_window(indexed_video, [0, 0, 1, 0, 1, 1, 0, 1], object())
    assert (window.start, window.end, window.confirmed) == (0, 899, True)


@pytest.mark.parametrize(
    "indexed_video,expected_end,confirmed",
    [(483, 482, True), (900, 899, False)],
    indirect=["indexed_video"],
)
def test_repeated_ocr_misses_require_a_blank_center(
    indexed_video, expected_end, confirmed, monkeypatch
):
    def recognize(engine, image):
        index = int(image[0, 0, 0]) + int(image[0, 0, 1]) * 256
        return ("PRESS HERE" if index < 120 else "PRSSHER"), 0.95

    monkeypatch.setattr(marker, "read_text", recognize)
    window = marker.find_test_window(indexed_video, [0, 0, 1, 0, 1, 1, 0, 1], object())
    assert (window.start, window.end, window.confirmed) == (0, expected_end, confirmed)
    assert any(c["state"] == "absent" and not c["center_blank"] for c in window.checks)


def test_no_marker_does_not_analyze_the_whole_video(indexed_video, monkeypatch):
    monkeypatch.setattr(marker, "read_text", lambda engine, image: ("HOME", 0.99))
    window = marker.find_test_window(indexed_video, [0, 0, 1, 0, 1, 1, 0, 1], object())
    assert window.start is None and window.end is None and not window.confirmed


def test_detector_uses_only_valid_range_and_keeps_original_frame_numbers():
    light, green = [255.0] * 800, [1.0] * 800
    light[200:600] = [180.0] * 400
    green[200:600] = [0.0] * 400
    for start in (200, 400):
        for index in (start + 20, start + 40):
            light[index : index + 3] = [100.0] * 3
        green[start + 50 : start + 100] = [0.8] * 50
    light[30:40] = [0.0] * 10
    result = detect_rounds(
        light, green, [0.8 if v == 0 else 0 for v in green], start_frame=200, stop_frame=600
    )
    assert len(result.rounds) == 2
    assert [r.green.frame for r in result.rounds] == [250, 450]
    assert result.diagnostics["flash_off_frames"] == [220, 240, 420, 440]
    assert result.diagnostics["analysis_range"] == {"start": 200, "end": 599}


def test_first_round_does_not_require_an_extra_reset_delay():
    light, green = [255.0] * 100, [0.0] * 100
    for start in (52, 56):
        light[start : start + 2] = [0.0] * 2
    green[60:85] = [0.8] * 25
    result = detect_rounds(
        light, green, [0.8 if v == 0 else 0 for v in green], start_frame=50, stop_frame=100
    )
    assert len(result.rounds) == 1 and result.rounds[0].green.frame == 60


def test_timestamp_errors_outside_valid_range_do_not_invalidate_measurement(session):
    session.test_start, session.test_end, session.range_confirmed = 5, 30, True
    session.method = "timestamps"
    session.video.times[80] = session.video.times[79]
    assert not any("不严格递增" in text for text in session.problems())
