import json
from dataclasses import asdict
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from pcts_video_tool.analysis import analyze
from pcts_video_tool.detection import FeatureExtractor, TextColorTracker
from pcts_video_tool.models import load_session, normalize_roi
from pcts_video_tool.run_logs import export_diagnostics
from pcts_video_tool.ui import VideoCanvas
from pcts_video_tool.video import crop


def test_old_rectangles_migrate_without_changing_frame_numbers(session, tmp_path):
    session.rois = {"green": [0.1, 0.2, 0.3, 0.4]}
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"format": 1, **asdict(session)}))
    loaded = load_session(path)
    assert loaded.rois["green"] == pytest.approx([0.1, 0.2, 0.4, 0.2, 0.4, 0.6, 0.1, 0.6])
    assert loaded.rounds[0].press == session.rounds[0].press


def test_crossed_and_tiny_regions_are_rejected():
    with pytest.raises(ValueError, match="交叉"):
        normalize_roi([0.1, 0.1, 0.8, 0.8, 0.8, 0.1, 0.1, 0.8], 640, 360)
    with pytest.raises(ValueError, match="太小"):
        normalize_roi([0.1, 0.1, 0.001, 0.001], 640, 360)


def test_polygon_measurement_excludes_outside_green_and_preserves_original_pixels():
    image = np.full((200, 300, 3), (0, 200, 0), np.uint8)
    roi = [0.2, 0.4, 0.7, 0.2, 0.8, 0.65, 0.3, 0.85]
    points = np.rint(np.array(roi).reshape(4, 2) * [299, 199]).astype(np.int32)
    cv2.fillConvexPoly(image, points, (220, 220, 220))
    measure = FeatureExtractor(300, 200, {"flash": roi, "green": roi})
    brightness, green, black = measure(image)
    assert brightness == 220 and green == 0 and black == 0


def test_clipped_highlights_do_not_hide_flash_change():
    roi = [0, 0, 1, 0, 1, 1, 0, 1]
    measure = FeatureExtractor(100, 100, {"flash": roi, "green": roi})
    on = np.full((100, 100, 3), 255, np.uint8)
    off = np.full_like(on, 150)
    off[:20] = 255  # The brightest 20% is saturated in both states.
    assert np.percentile(cv2.cvtColor(off, cv2.COLOR_BGR2GRAY), 85) == 255
    assert measure(on)[0] - measure(off)[0] > 70


def test_letter_color_is_not_the_colorful_background():
    reference = np.full((130, 360, 3), 230, np.uint8)
    cv2.putText(reference, "PRESS HERE", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    tracker = TextColorTracker(reference, [0, 0, 1, 0, 1, 1, 0, 1])
    background = np.full_like(reference, 230)
    cv2.circle(background, (80, 65), 60, (0, 190, 0), -1)
    cv2.circle(background, (270, 80), 50, (220, 80, 180), -1)
    dark = background.copy()
    cv2.putText(dark, "PRESS HERE", (33, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    green, black = tracker(dark)
    assert black > 0.5 and black > green + 0.2
    bright = background.copy()
    cv2.putText(bright, "PRESS HERE", (33, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 0), 2)
    green, black = tracker(bright)
    assert green > 0.25 and green > black + 0.04
    green, black = tracker(np.full_like(reference, (0, 220, 0)))
    assert green < 0.25 and black < 0.25  # No letter shape in a solid green region.


def test_perspective_crop_restores_slanted_text_plane():
    width, height = 320, 160
    source = np.full((height, width, 3), 255, np.uint8)
    source[30:60, 30:60] = (0, 0, 255)
    source[95:125, 250:285] = (255, 0, 0)
    points = np.float32([[80, 90], [420, 30], [450, 220], [110, 280]])
    transform = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [319, 0], [319, 159], [0, 159]]), points
    )
    skewed = cv2.warpPerspective(source, transform, (520, 320))
    roi = (points / [519, 319]).reshape(-1).tolist()
    restored = crop(skewed, normalize_roi(roi, 520, 320))
    red = restored[round(restored.shape[0] * 0.28), round(restored.shape[1] * 0.14)]
    blue = restored[round(restored.shape[0] * 0.69), round(restored.shape[1] * 0.84)]
    assert red[2] > 200 and red[0] < 20
    assert blue[0] > 200 and blue[2] < 20


def test_four_point_ui_respects_letterboxing_and_allows_corner_drag():
    app = QApplication.instance() or QApplication([])
    canvas = VideoCanvas()
    canvas.resize(800, 600)
    canvas.set_image(np.zeros((360, 640, 3), np.uint8))
    canvas.set_region("green")
    canvas.show()
    app.processEvents()
    received = []
    canvas.region_changed.connect(lambda name, roi: received.append((name, roi)))
    rect = canvas.image_rect()
    points = [(0.2, 0.3), (0.8, 0.2), (0.7, 0.8), (0.3, 0.9)]
    for x, y in points:
        QTest.mouseClick(
            canvas,
            Qt.LeftButton,
            pos=QPoint(round(rect.x() + x * rect.width()), round(rect.y() + y * rect.height())),
        )
    assert len(received) == 1
    assert received[0][1] == pytest.approx([v for p in points for v in p], abs=0.003)
    canvas.rois["green"] = received[0][1]
    old = QPoint(round(rect.x() + 0.2 * rect.width()), round(rect.y() + 0.3 * rect.height()))
    new = old + QPoint(12, -8)
    QTest.mousePress(canvas, Qt.LeftButton, pos=old)
    QTest.mouseMove(canvas, new)
    QTest.mouseRelease(canvas, Qt.LeftButton, pos=new)
    assert len(received) == 2 and received[1][1][0] > 0.2
    canvas.close()


def test_failed_analysis_is_automatically_recorded_and_exportable(session, tmp_path):
    with pytest.raises(ValueError, match="框选"):
        analyze(session)
    directory = Path(session.diagnostic_dir)
    assert json.loads((directory / "status.json").read_text())["status"] == "failed"
    assert "ValueError" in (directory / "error.txt").read_text()
    bundle = tmp_path / "diagnostics.zip"
    export_diagnostics(session, bundle)
    with ZipFile(bundle) as archive:
        assert "analysis/error.txt" in archive.namelist()
        assert "analysis/input.session.json" in archive.namelist()
        assert "current.session.json" in archive.namelist()
        assert not any(name.endswith((".mov", ".nut")) for name in archive.namelist())
