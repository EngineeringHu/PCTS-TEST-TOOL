import socket

import cv2
import numpy as np

from pcts_video_tool.recognition import create_hand_detector, create_ocr, fingertip, read_numbers


def test_pretrained_models_run_without_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("运行模型不应访问网络")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    image = np.full((120, 650, 3), 255, np.uint8)
    cv2.putText(image, "MD:88.1 TX:-0.7", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
    text, values, score = read_numbers(create_ocr(), image)
    assert values == (88.1, -0.7), text
    assert score > 0.8
    with create_hand_detector() as hand:
        blank = np.full((360, 640, 3), 255, np.uint8)
        assert fingertip(hand, blank, 0, [0, 0, 1, 0, 1, 1, 0, 1]) is None
