import cv2
import numpy as np
import pytest

from pcts_video_tool.detection import TextColorTracker, detect_rounds, refine_green_start

ROI = [0, 0, 1, 0, 1, 1, 0, 1]


def lettering(color=(0, 0, 0), background=(230, 230, 230), covered=False):
    image = np.full((120, 360, 3), background, np.uint8)
    cv2.putText(image, "PRESS HERE", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    if covered:
        image[:, 180:] = (110, 160, 210)
    return image


def proof(coarse=30, anchor=9):
    return {"last_black_frame": anchor, "first_green_candidate": coarse, "confirmed_at": coarse + 5}


def test_finds_earlier_letter_change_with_partial_occlusion_and_later_confirmation():
    tracker = TextColorTracker(lettering(), ROI)
    records = []

    def read_at(frame):
        return lettering((0, 180, 0) if frame >= 20 else (0, 0, 0), covered=True)

    result = refine_green_start(tracker, read_at, proof(), 240, record=records.append)
    assert result["candidate_frame"] == 20 and result["confirmed_at"] == 25
    assert any(
        row["kind"] == "sample" and row["frame"] == 10 and not row["passed"] for row in records
    )
    assert result["sector_pixels"][-1] < 8


@pytest.mark.parametrize(
    "early_scene", ["dark_green_letters", "green_background", "letters_removed"]
)
def test_does_not_use_interval_start_for_dark_tint_or_background_green(early_scene):
    tracker = TextColorTracker(lettering(), ROI)
    records = []

    def read_at(frame):
        if frame <= 9:
            return lettering()
        if frame >= 20:
            return lettering((0, 180, 0))
        if early_scene == "dark_green_letters":
            return lettering((25, 80, 25))
        if early_scene == "green_background":
            return lettering(background=(0, 180, 0))
        return np.full((120, 360, 3), (0, 180, 0), np.uint8)

    result = refine_green_start(tracker, read_at, proof(), 240, record=records.append)
    assert result["candidate_frame"] == 20
    assert all(
        not row["passed"] for row in records if row["kind"] == "sample" and 10 <= row["frame"] < 20
    )


def test_discards_isolated_early_green_and_keeps_first_frame_through_short_flicker():
    tracker = TextColorTracker(lettering(), ROI)

    def read_at(frame):
        green = frame == 11 or (frame >= 20 and frame != 21)
        return lettering((0, 180, 0) if green else (0, 0, 0))

    result = refine_green_start(tracker, read_at, proof(), 240)
    assert result["candidate_frame"] == 20 and result["confirmed_at"] == 26


def test_never_moves_existing_candidate_later_and_handles_missing_reference():
    tracker = TextColorTracker(lettering(), ROI)
    result = refine_green_start(
        tracker, lambda f: lettering((0, 180, 0) if f >= 21 else (0, 0, 0)), proof(coarse=20), 240
    )
    assert result["candidate_frame"] is None
    result = refine_green_start(tracker, lambda f: lettering(), proof(anchor=1), 240)
    assert result["candidate_frame"] is None and "不足三帧" in result["reason"]
    result = refine_green_start(
        tracker, lambda f: np.full((120, 360, 3), 230, np.uint8), proof(), 240
    )
    assert result["candidate_frame"] is None


def test_reference_must_have_spatial_support_and_cancel_keeps_partial_records():
    tracker = TextColorTracker(lettering(), ROI)

    def only_one_part(frame):
        image = lettering((0, 180, 0) if frame > 9 else (0, 0, 0))
        image[:, 100:] = (110, 160, 210)
        return image

    assert refine_green_start(tracker, only_one_part, proof(), 240)["candidate_frame"] is None
    records = []
    with pytest.raises(InterruptedError):
        refine_green_start(
            tracker,
            lambda f: lettering(),
            proof(),
            240,
            cancelled=lambda: len(records) >= 2,
            record=records.append,
        )
    assert len(records) == 2


@pytest.mark.parametrize("shift", [(1, 0), (3, 2)])
def test_shifted_dark_letters_do_not_expose_background_as_a_false_early_green(shift):
    tracker = TextColorTracker(lettering(), ROI)
    images = []
    for frame in range(40):
        if frame < 10:
            image = lettering()
        else:
            background = (0, 180, 0) if frame < 20 else (230, 230, 230)
            color = (25, 80, 25) if frame < 20 else (0, 180, 0)
            image = np.full((120, 360, 3), background, np.uint8)
            cv2.putText(
                image,
                "PRESS HERE",
                (25 + shift[0], 75 + shift[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                color,
                2,
            )
        images.append(image)
    scores = [tracker(image) for image in images]
    light = [255.0] * 40
    light[3] = light[6] = 0
    detected = detect_rounds(light, [s[0] for s in scores], [s[1] for s in scores])
    assert detected.rounds[0].green.frame == 20
    result = refine_green_start(
        tracker, lambda frame: images[frame], detected.diagnostics["green_evidence"][0], 240
    )
    assert (
        result["candidate_frame"] is None
    )  # Never revise the correct 20 to background exposure at 10.
