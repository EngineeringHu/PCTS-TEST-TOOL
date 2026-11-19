from copy import deepcopy
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from pcts_video_tool.detection import TextColorTracker, detect_rounds, pair_responses
from pcts_video_tool.gray_recovery import missing_windows, recover_gray_rounds, retain_manual_fields
from pcts_video_tool.models import Event, Round
from pcts_video_tool.video import inspect_video

ROI = [0, 0, 1, 0, 1, 1, 0, 1]


def test_adaptive_reference_requires_recoverable_gray_strokes():
    ref = np.full((80, 240, 3), 220, np.uint8)
    assert TextColorTracker.from_gray_reference(ref, ROI) is None
    cv2.putText(ref, "PRESS HERE", (6, 55), 0, 1.2, (150, 150, 150), 3)
    tracker = TextColorTracker.from_gray_reference(ref, ROI)
    assert tracker is not None and tracker.gray_reference["threshold"] == 150
    assert tracker.gray_reference["adaptive_ink_pixels"] > 1500
    with pytest.raises(ValueError):
        TextColorTracker(ref, ROI)
    for color in ((0, 0, 0), (0, 170, 0)):
        ref[:] = 220
        cv2.putText(ref, "PRESS HERE", (6, 55), 0, 1.2, color, 3)
        assert TextColorTracker.from_gray_reference(ref, ROI) is None


@pytest.mark.parametrize(
    "offs,black_count,green_count,pairs",
    [
        ([18, 25], 18, 8, 1),
        ([18], 18, 8, 0),
        ([16, 18, 25], 18, 8, 0),
        ([18, 25], 2, 8, 0),
        ([18, 25], 18, 3, 0),
        ([18, 30], 18, 8, 0),
        ([18, 25], 18, 0, 0),
    ],
)
def test_recovery_requires_black_reset_exact_pair_and_sustained_green(
    offs, black_count, green_count, pairs
):
    g, b = np.zeros(80), np.zeros(80)
    b[:black_count] = 0.7
    g[30 : 30 + green_count] = 0.7
    assert len(pair_responses(offs, g, b, 240, 0, 80, strict_pairs=True).rounds) == pairs


def test_rescue_uses_only_gaps_and_preserves_the_existing_pair(
    generated_video, monkeypatch, tmp_path
):
    path, rois = generated_video
    info = inspect_video(str(path))
    # Three real responses in the manual interval. Simulate a fragmented template which only saw the middle one.
    lights, g, b = np.full(300, 255.0), np.zeros(300), np.ones(300)
    for q in (20, 40, 120, 140, 220, 240):
        lights[q : q + 4] = 0
    g[150:180] = 0.9
    b[150:180] = 0
    first = detect_rounds(lights, g, b, 240, 8, 295)
    retained = deepcopy(first.rounds[0])
    assert missing_windows(first, 8, 295, 240) == [(8, 120, [20, 40]), (196, 295, [220, 240])]

    # Use the known full glyph to exercise the recovery pass on actual decoded frames.
    def full_tracker(ref, roi):
        tracker = TextColorTracker(ref, roi)
        tracker.gray_reference = {"threshold": 150}
        return tracker

    monkeypatch.setattr(TextColorTracker, "from_gray_reference", full_tracker)
    _, added = recover_gray_rounds(
        first,
        info,
        rois["green"],
        8,
        SimpleNamespace(start=8, end=294),
        240,
        tmp_path,
        lambda *_: None,
        lambda: False,
    )
    assert list(added) == [50, 250]
    assert [r.green.frame for r in first.rounds] == [50, 150, 250]
    assert first.rounds[1] == retained and not first.diagnostics["unmatched_flash_frames"]
    assert first.diagnostics["gray_recovery"]["matched_frames"] == 211
    assert (tmp_path / "gray-template.png").is_file()


def test_recovery_keeps_manual_edits_with_their_evidence_after_renumbering():
    rounds = [
        Round(blink1=Event(10), blink2=Event(20), green=Event(30), last_green_frame=50),
        Round(blink1=Event(60), blink2=Event(70), green=Event(80), last_green_frame=99),
    ]
    old = deepcopy(rounds[-1])
    old.press = Event(55, "manual", "触屏已核对")
    old.md, old.tx, old.reading_frame = 81.2, 0.6, 92
    old.set_reading_state("md", "manual")
    retain_manual_fields([old], rounds, 0)
    assert rounds[1].press == old.press and rounds[0].press.frame is None
    assert (rounds[1].md, rounds[1].tx, rounds[1].reading_frame) == (81.2, 0.6, 92)
    assert rounds[1].reading_states == old.reading_states
    with pytest.raises(ValueError, match="无法唯一对应"):
        retain_manual_fields([old, old], rounds, 0)


def test_unusable_initial_template_uses_one_scan(generated_video, tmp_path, monkeypatch):
    from fractions import Fraction

    import av

    from pcts_video_tool import analysis
    from pcts_video_tool.models import Session

    original, rois = generated_video
    path = tmp_path / "pale-reference.nut"
    with av.open(str(original)) as source, av.open(str(path), "w") as out:
        stream = out.add_stream("ffv1", rate=240)
        stream.width, stream.height, stream.pix_fmt = 640, 360, "bgr0"
        for index, frame in enumerate(source.decode(video=0)):
            if index == 100:
                break
            picture = frame.to_ndarray(format="bgr24")
            if index == 0:
                picture = np.maximum(picture, 150)
            sample = av.VideoFrame.from_ndarray(picture, format="bgr24")
            sample.pts, sample.time_base = index, Fraction(1, 240)
            for packet in stream.encode(sample):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    monkeypatch.setattr(analysis, "create_ocr", lambda: None)
    monkeypatch.setattr(analysis, "create_hand_detector", lambda **kw: None)
    monkeypatch.setattr(analysis, "recover_gray_rounds", lambda *args: pytest.fail("重复补扫"))
    result = analysis.analyze(
        Session(
            inspect_video(str(path)),
            rois,
            range_mode="manual",
            test_start=0,
            test_end=99,
            range_confirmed=True,
        )
    )
    assert [(r.blink1.frame, r.blink2.frame, r.green.frame) for r in result.rounds] == [
        (20, 40, 50)
    ]
