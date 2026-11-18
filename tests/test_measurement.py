import json

import pytest

from pcts_video_tool.models import calculate, load_session, save_session
from pcts_video_tool.video import sampling_fps


def test_switch_time_policy_preserves_inputs(session):
    frames = calculate(session, session.rounds[0])
    assert frames["end_to_end"] == pytest.approx(50)
    assert frames["md_total"] == 11
    assert frames["hu_total"] == pytest.approx(39)
    original = session.rounds[0].press.frame
    session.method = "timestamps"
    timed = calculate(session, session.rounds[0])
    assert timed["end_to_end"] == pytest.approx(58.3333333333)
    assert session.rounds[0].press.frame == original
    assert timed["md_total"] == frames["md_total"]


def test_missing_values_are_not_zero(session):
    item = session.rounds[0]
    item.press.frame = None
    item.tx = None
    result = calculate(session, item)
    assert result["end_to_end"] is None
    assert result["md_total"] is None
    assert result["hu_output"] is not None


def test_missing_timestamp_does_not_use_fps(session):
    session.method = "timestamps"
    session.video.times[10] = None
    assert calculate(session, session.rounds[0])["end_to_end"] is None
    assert any("缺少时间戳" in issue for issue in session.problems())


def test_uniform_shift_preserves_frame_calculation(session):
    original = calculate(session, session.rounds[0])
    for event in (
        session.rounds[0].press,
        session.rounds[0].blink1,
        session.rounds[0].blink2,
        session.rounds[0].green,
    ):
        event.frame += 3
    assert calculate(session, session.rounds[0]) == original


def test_session_roundtrip_and_external_validation(session, tmp_path):
    path = tmp_path / "task.json"
    save_session(session, path)
    restored = load_session(path)
    assert restored == session
    raw = json.loads(path.read_text())
    raw["rounds"][0]["press"]["frame"] = 9999
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="超出"):
        load_session(path)


def test_incorrect_order_and_round_count_are_visible(session):
    session.rounds[0].press.frame = 90
    assert any("顺序" in text for text in session.problems())
    assert any("当前1轮" in text for text in session.problems())


def test_detection_rate_uses_pts_spacing_not_container_base_rate(session):
    session.video.nominal_fps = 90000
    session.fps = 60
    assert sampling_fps(session.video) == pytest.approx(240)
    assert session.fps == 60  # User's measurement policy is independent.
