import json

import cv2
import numpy as np
import pytest

from pcts_video_tool.models import TouchCalibration, load_session, save_session
from pcts_video_tool.motion import (
    FingerObservation,
    MotionTracker,
    locate_motion_press,
    prepare_motion_tracker,
)


def observations(positions, angle=0, offset=(0, 0), missing=()):
    axis = np.array([np.cos(angle), np.sin(angle)])
    across = np.array([-axis[1], axis[0]])
    result = []
    for frame, position in enumerate(positions):
        if frame in missing:
            result.append(FingerObservation(frame))
            continue
        center = np.array([100, 100]) + offset + position * axis
        points = [
            center + a * axis + b * across for a, b in ((-15, -8), (15, -8), (15, 8), (-15, 8))
        ]
        mask = np.zeros((250, 250), np.uint8)
        cv2.fillConvexPoly(mask, np.rint(points).astype(np.int32), 1)
        result.append(FingerObservation(frame, mask, 0.9))
    return result


@pytest.mark.parametrize("angle,offset", [(0, (0, 0)), (0.8, (20, -10)), (2.1, (-20, 30))])
def test_motion_onset_does_not_require_absolute_line_or_position(angle, offset):
    positions = list(range(0, 31, 3)) + [30] * 10 + list(range(27, -1, -3))
    event, decision, rows = locate_motion_press(
        observations(positions, angle, offset), 240, 20, np.zeros(2)
    )
    assert abs(event.frame - 10) <= 1
    assert event.state == "review" and "物理接触" in event.reason
    assert decision["review_interval"][0] <= event.frame <= decision["review_interval"][1]
    assert rows[event.frame]["x"] > 0


@pytest.mark.parametrize("latest", [2, 3, 8])
def test_short_visible_approach_can_confirm_using_later_hold_without_dating_it_later(latest):
    values = observations([0, 5, 10] + [10] * 10)
    event, decision, _ = locate_motion_press(values, 240, latest, np.zeros(2))
    assert event.frame == 2
    assert decision["candidates"][0]["confirmed_at"] > 2


def test_hover_followed_by_clear_further_approach_is_not_first_press():
    positions = list(range(0, 16, 3)) + [15] * 7 + list(range(18, 43, 3)) + [42] * 8
    event, decision, _ = locate_motion_press(
        observations(positions), 240, len(positions) - 1, np.zeros(2)
    )
    assert event.frame >= 20
    assert 5 in decision["rejected_pauses"]


def test_static_missing_and_after_blink_do_not_fabricate_a_press():
    for positions, missing, latest in [
        ([20] * 20, (), 19),
        (list(range(0, 31, 3)) + [30] * 8, (8, 9, 10, 11, 12), 18),
        (list(range(0, 31, 3)) + [30] * 8, (), 5),
    ]:
        event, _, _ = locate_motion_press(
            observations(positions, missing=missing), 240, latest, np.zeros(2)
        )
        assert event.frame is None


def picture(x=80, absent=False):
    image = np.full((100, 180, 3), 240, np.uint8)
    if not absent:
        cv2.rectangle(image, (x, 35), (x + 30, 65), (100, 150, 200), -1)
    return image


def test_tracking_stops_after_loss_and_does_not_reacquire_same_colour_background():
    roi = [0, 0, 1, 0, 1, 1, 0, 1]
    tracker = MotionTracker(picture(), roi, [90 / 179, 50 / 99])
    patches = [(i, picture(80 if i < 4 else 25, absent=i in (4, 5))) for i in range(10)]
    result = tracker.trace(patches, 2)
    assert all(o.mask is not None for o in result[:4])
    assert all(o.mask is None for o in result[4:])


def test_sample_mode_uses_appearance_frame_not_contact_position():
    roi = [0, 0, 1, 0, 1, 1, 0, 1]
    tracker, anchor, report = prepare_motion_tracker(
        lambda f: picture(80 + (f % 4)),
        roi,
        0,
        110,
        100,
        240,
        sample=TouchCalibration(25, [], [90 / 179, 50 / 99]),
    )
    assert tracker is not None and anchor is not None
    assert report["source"] == "manual_sample"
    sampled = [item["frame"] for item in report["attempts"]]
    assert 25 in sampled  # Include the user's known-visible frame in its own round.
    assert 0 in sampled and 50 in sampled and 100 in sampled  # No assumed press-to-blink delay.


def test_sample_persistence_and_old_tasks_default_to_auto(session, tmp_path):
    session.rois["touch"] = [0, 0, 1, 0, 1, 1, 0, 1]
    session.press_mode = "sample"
    session.touch_calibration = TouchCalibration(20, [], [0.6, 0.5])
    path = tmp_path / "session.json"
    save_session(session, path)
    assert load_session(path) == session
    raw = json.loads(path.read_text())
    raw.update(format=4)
    raw.pop("press_mode")
    raw["touch_calibration"]["line"] = [0.4, 0.2, 0.4, 0.8]
    path.write_text(json.dumps(raw))
    assert load_session(path).press_mode == "auto"
    assert load_session(path).touch_calibration is not None
    raw["press_mode"] = "reference"
    raw["touch_calibration"]["line"] = []
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="参考线"):
        load_session(path)


@pytest.mark.parametrize("flow_available", [True, False])
def test_roi_edge_does_not_turn_constant_speed_into_a_press(flow_available):
    values = []
    old_area = 0
    for i in range(36):
        mask = np.zeros((100, 100), np.uint8)
        y = -20 + 3 * i
        cv2.rectangle(mask, (40, y), (60, y + 30), 1, -1)
        area = int(mask.sum())
        values.append(
            FingerObservation(
                i,
                mask,
                0.9,
                flow_points=10 if flow_available else 0,
                flow_y=3,
                area_change=area - old_area,
            )
        )
        old_area = area
    event, diagnostic, _ = locate_motion_press(values, 240, 34, np.zeros(2))
    assert event.frame is None
    assert diagnostic["rejected_clipping"] or diagnostic["rejected_moving"]


def test_reliable_texture_motion_vetoes_false_static_contour():
    values = observations(list(range(0, 31, 3)) + [30] * 10)
    for o in values:
        o.flow_points, o.flow_x, o.flow_y = 10, 3, 0
    event, diagnostic, _ = locate_motion_press(values, 240, 20, np.zeros(2))
    assert event.frame is None and diagnostic["rejected_moving"]
