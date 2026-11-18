import json
from dataclasses import replace

import cv2
import numpy as np
import pytest

from pcts_video_tool.models import (
    TouchCalibration,
    load_session,
    save_session,
    validate_touch_calibration,
)
from pcts_video_tool.touch import ContactTracker, TouchSample, locate_contact


def calibration():
    return TouchCalibration(10, [100 / 199, 20 / 119, 100 / 199, 100 / 119], [130 / 199, 60 / 119])


def finger(gap=0, width=22):
    image = np.full((120, 200, 3), 240, np.uint8)
    cv2.rectangle(
        image, (100 + gap, 60 - width // 2), (150 + gap, 60 + width // 2), (100, 150, 200), -1
    )
    return image


def test_original_pixel_front_distance_and_missing_contour():
    tracker = ContactTracker(finger(), [0.1, 0.1, 0.95, 0.1, 0.95, 0.9, 0.1, 0.9], calibration())
    assert tracker.measure(1, finger(12)).gap_px == pytest.approx(12, abs=0.1)
    assert tracker.measure(2, finger()).gap_px == pytest.approx(0, abs=0.1)
    assert tracker.measure(3, np.full((120, 200, 3), 240, np.uint8)).gap_px is None
    reversed_line = replace(calibration(), line=[100 / 199, 100 / 119, 100 / 199, 20 / 119])
    reverse = ContactTracker(finger(), [0.1, 0.1, 0.95, 0.1, 0.95, 0.9, 0.1, 0.9], reversed_line)
    assert reverse.measure(1, finger(12)).gap_px == pytest.approx(12, abs=0.1)


def samples(gaps):
    return [
        TouchSample(i) if gap is None else TouchSample(i, gap, 100.0, 60.0, 12.0, 200, 0.9)
        for i, gap in enumerate(gaps)
    ]


def test_later_hold_confirms_but_does_not_move_first_arrival_frame():
    values = samples([18, 16, 14, 12, 10, 8, 6, 4, 3, 2, 0, 0, 0, 0, 4, 8])
    event, record = locate_contact(values, 240, latest_press=11)
    assert event.frame == 10 and event.state == "review"
    assert record["candidates"][0]["confirmed_at"] == 12  # May confirm after blink1.
    assert "物理接触" in event.reason


def test_waiting_for_later_stop_returns_start_of_continuous_arrival_before_blink():
    values = samples([18, 16, 14, 12, 10, 8, 6, 4, 3, 2, 1.5, 0.5, -0.5, -0.5, -0.5])
    event, record = locate_contact(values, 240, latest_press=10)
    assert event.frame == 10
    assert record["candidates"][0]["confirmed_at"] == 13


def test_hover_and_single_bad_pixel_do_not_become_contact():
    for gaps in ([18, 14, 10, 6, 4, 4, 4, 4, 4, 8], [18, 14, 10, 6, 0, 6, 6, 6]):
        event, _ = locate_contact(samples(gaps), 240, latest_press=len(gaps) - 1)
        assert event.frame is None


def test_contour_widening_can_confirm_arrival_before_motion_fully_stops():
    values = samples([12, 10, 8, 6, 4, 0, -1.2, -2.4])
    for sample in values[5:]:
        sample.front_width_px = 18
    event, record = locate_contact(values, 240, latest_press=6)
    assert event.frame == 5
    assert record["candidates"][0]["deformation"]
    assert not record["candidates"][0]["stopped"]


def test_missing_contour_stays_unknown_and_reference_arrival_requires_approach():
    event, record = locate_contact(samples([12, 10, 8, 6, 4, None, None, 0, 0, 0]), 240, 8)
    assert event.frame == 7 and "不可观察" in event.reason
    assert record["observed_frames"] == 8
    assert locate_contact(samples([0] * 10), 240, 9)[0].frame is None
    assert locate_contact(samples([12, 10, 8, 6, 4, 0, 0, 0]), 240, 4)[0].frame is None


def test_calibration_roundtrip_old_format_and_boundary_validation(session, tmp_path):
    session.rois["touch"] = [0, 0, 1, 0, 1, 1, 0, 1]
    session.touch_calibration = TouchCalibration(20, [0.4, 0.2, 0.4, 0.8], [0.6, 0.5])
    path = tmp_path / "calibrated.json"
    save_session(session, path)
    assert load_session(path) == session
    raw = json.loads(path.read_text())
    assert raw["format"] == 4
    raw["format"] = 3
    raw.pop("touch_calibration")
    path.write_text(json.dumps(raw))
    assert load_session(path).touch_calibration is None
    with pytest.raises(ValueError, match="取样点"):
        validate_touch_calibration(
            replace(session.touch_calibration, skin_point=[0.4, 0.5]),
            session.video,
            session.rois["touch"],
        )
    with pytest.raises(ValueError, match="太短"):
        validate_touch_calibration(
            replace(session.touch_calibration, line=[0.4, 0.2, 0.4, 0.2]),
            session.video,
            session.rois["touch"],
        )
