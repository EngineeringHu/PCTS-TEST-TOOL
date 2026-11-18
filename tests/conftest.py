from fractions import Fraction

import av
import cv2
import numpy as np
import pytest

from pcts_video_tool.models import Event, Round, Session, VideoInfo, normalize_roi


@pytest.fixture(autouse=True)
def diagnostic_home(monkeypatch, tmp_path):
    monkeypatch.setenv("PCTS_LOG_DIR", str(tmp_path / "diagnostics"))


@pytest.fixture
def session():
    times = [i / 240 for i in range(100)]
    for i in range(18, 100):
        times[i] += 2 / 240
    info = VideoInfo("unused", 640, 360, 0, 0, times, 240, 240)
    item = Round(
        *(Event(i, "manual", "") for i in (10, 14, 20, 22)),
        md=11.5,
        tx=-0.5,
        reading_frame=25,
        reading_state="manual",
    )
    return Session(info, rounds=[item])


@pytest.fixture(scope="session")
def generated_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("synthetic") / "independent-ten-rounds.nut"
    rois = {
        "flash": [0.025, 0.025, 0.125, 0.2],
        "green": [0.4, 0.1, 0.55, 0.2],
        "reading": [0.32, 0.5, 0.65, 0.28],
        "touch": [0, 0, 1, 1],
    }
    rois = {name: normalize_roi(value, 640, 360) for name, value in rois.items()}
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=240)
        stream.width, stream.height = 640, 360
        stream.pix_fmt = "bgr0"
        stream.time_base = Fraction(1, 2400)
        for index in range(1000):
            phase = index % 100
            image = np.full((360, 640, 3), 230, np.uint8)
            light = 0 if 20 <= phase < 24 or 40 <= phase < 44 else 255
            image[9:81, 16:96] = light
            is_green = 50 <= phase < 80
            cv2.putText(
                image,
                "PRESS HERE",
                (270, 83),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 170, 0) if is_green else (0, 0, 0),
                2,
            )
            if 53 <= phase < 80:
                cv2.putText(
                    image,
                    "MD:88.1 TX:-0.7",
                    (215, 245),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 120, 0),
                    2,
                )
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts = index * 10 + (20 if index >= 500 else 0)
            frame.time_base = Fraction(1, 2400)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path, rois
