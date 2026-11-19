import cv2
import numpy as np
import pytest

from pcts_video_tool.green_recommendation import GreenOnsetProbe, interior_glyph


@pytest.mark.parametrize(
    "scene,baseline,expected",
    [
        ("letters", 26, 20),
        ("letters", 25, None),
        ("background", 40, None),
        ("single sector", 40, None),
        ("one frame", 40, None),
    ],
)
def test_recommendation_requires_a_sustained_change_on_two_stroke_regions(
    scene, baseline, expected
):
    glyph = np.zeros((60, 150), np.uint8)
    glyph[:, :3] = 255  # A dark bezel touches the template boundary.
    cv2.putText(glyph, "PRESS HERE", (12, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.65, 255, 2)
    cleaned = interior_glyph(glyph)
    assert not cleaned[:, :3].any()
    probe = GreenOnsetProbe(cleaned, (0, 0, 1), 10, baseline)
    for index in range(probe.start, baseline):
        if index > probe.stop:
            break
        patch = np.full((*glyph.shape, 3), 225, np.uint8)
        patch[glyph > 0] = 0
        if index >= 20:
            if scene == "background":
                patch[glyph == 0] = (0, 180, 0)
            elif scene == "single sector":
                patch[:, :50][glyph[:, :50] > 0] = (0, 180, 0)
            elif scene != "one frame" or index == 20:
                patch[glyph > 0] = (0, 180, 0)
        probe.observe(index, patch)
    assert probe.candidate == expected
