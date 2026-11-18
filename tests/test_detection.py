import numpy as np

from pcts_video_tool.detection import FeatureExtractor, TextColorTracker, detect_rounds, stable_runs
from pcts_video_tool.models import Round
from pcts_video_tool.recognition import choose_reading, parse_reading, touch_candidate
from pcts_video_tool.video import inspect_video, read_frame


def test_actual_decode_detects_ten_rounds_and_exact_boundaries(generated_video):
    path, rois = generated_video
    light, green, black = [], [], []
    extract = FeatureExtractor(640, 360, rois)
    tracker = None

    def sample(i, image):
        nonlocal tracker
        if tracker is None:
            tracker = TextColorTracker(image, rois["green"])
        a, _, _ = extract(image)
        b, c = tracker(image)
        light.append(a)
        green.append(b)
        black.append(c)

    info = inspect_video(str(path), sample=sample)
    result = detect_rounds(light, green, black)
    assert len(info.times) == 1000
    assert info.average_fps < 240  # The generated time axis contains one long interval.
    assert len(result.rounds) == 10
    for i, item in enumerate(result.rounds):
        assert item.blink1.frame == 100 * i + 20
        assert item.blink2.frame == 100 * i + 40
        assert item.green.frame == 100 * i + 50
    for index in (0, 20, 40, 50, 550, 999, 20):
        image = read_frame(info, index)
        a, _, _ = extract(image)
        b, c = tracker(image)
        assert a == light[index] and b == green[index] and c == black[index]
    assert any("长间隔" in w for w in info.warnings)


def test_missing_flash_is_preserved_for_review():
    light = [255.0] * 200
    green = [0.0] * 200
    light[10:20] = [0.0] * 10
    green[50:80] = [0.5] * 30
    result = detect_rounds(light, green, [0.8 if v == 0 else 0 for v in green])
    assert len(result.rounds) == 1
    assert result.rounds[0].blink2.frame is None
    assert result.rounds[0].issues


def test_brief_flash_off_phases_below_five_percent_are_detected():
    light, green = [255.0] * 1000, [0.0] * 1000
    for i in range(10):
        light[100 * i + 20 : 100 * i + 22] = [0.0, 0.0]
        light[100 * i + 40 : 100 * i + 42] = [0.0, 0.0]
        green[100 * i + 50 : 100 * i + 80] = [0.5] * 30
    result = detect_rounds(light, green, [0.8 if v == 0 else 0 for v in green])
    assert len(result.rounds) == 10
    assert all(
        r.blink1.frame == 100 * i + 20 and r.blink2.frame == 100 * i + 40
        for i, r in enumerate(result.rounds)
    )


def test_no_green_does_not_discard_flash_pair():
    light = [255.0] * 100
    light[10:20] = [0.0] * 10
    light[30:40] = [0.0] * 10
    result = detect_rounds(light, [0.0] * 100, [0.8] * 100)
    assert not result.rounds
    assert result.diagnostics["unmatched_flash_frames"] == [10, 30]


def test_missing_green_preserves_earlier_and_later_pairs():
    light, green = [255.0] * 200, [0.0] * 200
    for frame in (20, 40, 120, 140):
        light[frame : frame + 2] = [0.0, 0.0]
    green[150:180] = [0.5] * 30
    detected = detect_rounds(light, green, [0.8 if v == 0 else 0 for v in green])
    assert len(detected.rounds) == 1
    item = detected.rounds[0]
    assert (item.blink1.frame, item.blink2.frame, item.green.frame) == (120, 140, 150)
    assert detected.diagnostics["unmatched_flash_frames"] == [20, 40]
    assert item.issues


def test_signed_decimals_and_reading_conflict():
    assert parse_reading("MD:88.1 TX:−0.7") == (88.1, -0.7)
    assert parse_reading("MD:103.2TX:0.6 PRESS HERE") == (103.2, 0.6)
    assert parse_reading("MD:88.1TX:-0.7") == (88.1, -0.7)
    assert parse_reading("MD:8O.1TX:0.7") is None
    assert parse_reading("MD:88.1") is None
    assert parse_reading("MD:8O.1 TX:0.7") is None
    assert parse_reading("MD:88.1 TX:3O.0") is None
    item = Round()
    choose_reading(
        item,
        [
            (5, "MD:88.1 TX:-0.7", (88.1, -0.7), 0.99),
            (6, "MD:88.1 TX:-0.7", (88.1, -0.7), 0.99),
            (7, "MD:88.1 TX:-8.7", (88.1, -8.7), 0.99),
        ],
    )
    assert item.reading_state == "review"
    assert item.tx == -0.7


def test_motion_candidate_is_not_claimed_as_contact():
    points = [(i, i * 0.01 if i < 6 else 0.06, 0.5) for i in range(12)]
    result = touch_candidate(points)
    assert result.frame is not None
    assert result.state == "review"
    assert touch_candidate([]).frame is None


def test_79_fragmented_green_segments_are_ten_responses():
    light, green = [255.0] * 10000, [0.0] * 10000
    for i in range(10):
        base = i * 1000
        for frame in (base + 120, base + 160):
            light[frame : frame + 3] = [0.0] * 3
        green[base + 200 : base + 800] = [0.8] * 600
        for j in range(7 if i < 9 else 6):
            start = base + 250 + j * 70
            green[start : start + 6] = [0.0] * 6
    assert len(stable_runs(np.asarray(green) > 0.01)) == 79
    result = detect_rounds(light, green, [0.8 if v == 0 else 0 for v in green])
    assert len(result.rounds) == 10
    assert [r.green.frame for r in result.rounds] == [i * 1000 + 200 for i in range(10)]
    assert result.green_ends == [i * 1000 + 800 for i in range(10)]


def test_isolated_flash_noise_does_not_manufacture_rounds_or_truncate_to_ten():
    light, green = [255.0] * 2400, [0.0] * 2400
    for i in range(12):
        base = i * 200
        for frame in (base + 20, base + 40, base + 60, base + 80):
            light[frame : frame + 2] = [0.0] * 2
        green[base + 100 : base + 160] = [0.8] * 60
    result = detect_rounds(light, green, [0.8 if v == 0 else 0 for v in green])
    assert len(result.rounds) == 12
    assert len(result.diagnostics["unmatched_flash_frames"]) == 24
    assert all(r.issues for r in result.rounds)


def test_absent_green_without_black_text_does_not_rearm():
    light, green, black = [255.0] * 500, [0.0] * 500, [0.8] * 500
    for frame in (40, 70):
        light[frame : frame + 3] = [0.0] * 3
    green[100:400] = [0.8] * 300
    black[100:400] = [0.0] * 300
    green[200:240] = [0.0] * 40  # An opaque finger hides the text, it did not return to black.
    result = detect_rounds(light, green, black)
    assert len(result.rounds) == 1
    assert result.green_ends == [400]


def test_first_green_frame_is_not_delayed_by_confirmation_or_one_frame_flicker():
    light, green = [255.0] * 200, [0.0] * 200
    for frame in (20, 40):
        light[frame : frame + 3] = [0.0] * 3
    green[50:100] = [0.8] * 50
    green[51] = 0
    result = detect_rounds(light, green, [0.8 if v == 0 else 0 for v in green])
    assert len(result.rounds) == 1
    assert result.rounds[0].green.frame == 50


def test_single_frame_flash_and_long_dark_state_both_keep_their_first_frame():
    light, green, black = [210.0] * 120, [0.0] * 120, [0.8] * 120
    light[10:35] = [150.0] * 25
    light[37] = 150.0
    green[60:95] = [0.8] * 35
    black[60:95] = [0.0] * 35
    result = detect_rounds(light, green, black)
    assert result.diagnostics["flash_off_frames"] == [10, 37]
    assert len(result.rounds) == 1
