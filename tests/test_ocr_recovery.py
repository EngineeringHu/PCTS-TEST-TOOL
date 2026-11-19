from types import SimpleNamespace

import numpy as np
import pytest

from pcts_video_tool.models import Round
from pcts_video_tool.recognition import (
    choose_reading,
    parse_reading,
    read_numbers,
    supplement_reading,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("MD:103.2TX:0.6", (103.2, 0.6)),
        ("MD:0 TX:-1", (0, -1)),
        ("MD：88.1 TX：−0.7", (88.1, -0.7)),
        ("MD:753 TX:6", (753, 6)),  # Integers are allowed; do not insert decimals.
        ("MD:88.12 TX:0.7", None),
        ("MD:88.1 TX:0.75", None),
        ("MD88.1 TX:0.7", None),
        ("MD:88.1 TX0.7", None),
        ("MD:88.1 TX:0.7O", None),
        ("MD:8O.1TX:0.7", None),
        ("MD:88.1 TX:0.", None),
    ],
)
def test_complete_format_with_at_most_one_decimal(text, expected):
    assert parse_reading(text) == expected


def test_unrelated_text_does_not_lower_numeric_line_score():
    def engine(image):
        return SimpleNamespace(txts=["MD:88.1TX:-0.7", "DRESS HEE"], scores=[0.98, 0.5])

    text, value, score = read_numbers(engine, np.zeros((100, 400, 3), np.uint8))
    assert value == (88.1, -0.7) and score == 0.98
    assert "DRESS HEE" in text  # Keep the original OCR evidence.


def test_labels_and_numbers_split_into_ocr_boxes_still_form_a_reading():
    def engine(image):
        return SimpleNamespace(txts=["MD:", "88.1", "TX:-0.7"], scores=[0.98, 0.92, 0.97])

    _, value, score = read_numbers(engine, np.zeros((100, 400, 3), np.uint8))
    assert value == (88.1, -0.7) and score == 0.92


def test_failed_seeds_search_both_directions_and_stop_on_independent_agreement():
    item = Round()
    readings = [(f, "MD88.12 TX0.7", None, 0.99) for f in (5, 10, 15)]
    visited = []

    def read_at(frame):
        visited.append(frame)
        return "MD:88.1 TX:-0.7", (88.1, -0.7), 0.99

    supplement_reading(item, readings, 0, 20, read_at)
    assert visited == [4, 6, 9, 11]
    assert (item.md, item.tx, item.reading_state) == (88.1, -0.7, "auto")
    assert len({f for f, *_ in readings}) == len(readings)


def test_neighbors_stay_in_current_round_and_do_not_repeat_frames():
    item = Round()
    readings = [(f, "", None, 0.0) for f in (100, 102, 105)]
    visited = []

    def read_at(frame):
        assert 100 <= frame < 106
        visited.append(frame)
        return "MD:88.12 TX:0.7", None, 0.99

    supplement_reading(item, readings, 100, 106, read_at)
    assert set(visited) == {101, 103, 104}
    assert len(visited) == len(set(visited))
    assert item.md is None and item.reading_state == "review"


def test_same_frame_does_not_count_as_multiple_votes_and_conflicts_remain_review():
    item = Round()
    row = (5, "MD:88.1 TX:0.7", (88.1, 0.7), 0.99)
    choose_reading(item, [row, row, row])
    assert item.reading_state == "review"
    readings = [row, (10, "MD:881 TX:7", (881, 7), 0.99), (15, "", None, 0.0)]
    supplement_reading(item, readings, 0, 50, lambda f: ("MD:88.1 TX:0.7", (88.1, 0.7), 0.99))
    assert (item.md, item.tx) == (88.1, 0.7)
    assert item.reading_state == "review" and len(readings) <= 51
    assert "MD:881 TX:7" in item.reading_reason


def test_search_covers_other_times_in_round_after_nearby_frames_fail():
    item = Round()
    readings = [(f, "", None, 0.0) for f in (60, 100, 140)]
    visited = []

    def read_at(frame):
        visited.append(frame)
        if frame in (10, 20, 30):
            return "MD:92.1 TX:0.8", (92.1, 0.8), 0.99
        return "MD:921 TX0.8", None, 0.9

    supplement_reading(item, readings, 0, 150, read_at)
    assert {10, 20, 30}.issubset(visited)
    assert (item.md, item.tx, item.reading_state) == (92.1, 0.8, "auto")
    assert len(visited) <= 48


@pytest.mark.parametrize("conflict", [False, True])
def test_sorted_prefetch_preserves_ocr_order_early_stop_and_votes(conflict):
    seeds = [(f, "", None, 0.0) for f in (22, 46, 63)]

    def reading(frame):
        md = 90.0 if conflict and frame == 21 else 91.0
        return f"MD:{md} TX:0.7", (md, 0.7), 0.99

    expected, expected_reads = Round(), seeds.copy()
    supplement_reading(expected, expected_reads, 12, 77, reading)
    assert expected.reading_state == ("review" if conflict else "auto")

    crops = {}

    def prefetch(frames):
        assert frames == sorted(set(frames))
        assert all(12 <= f < 77 for f in frames)
        crops.clear()
        crops.update((frame, frame) for frame in frames)

    actual, actual_reads = Round(), seeds.copy()
    supplement_reading(
        actual, actual_reads, 12, 77, lambda f: reading(crops.pop(f)), prefetch=prefetch
    )
    assert (actual, actual_reads) == (expected, expected_reads)


def test_neighbor_search_cancellation_keeps_collected_evidence():
    item = Round()
    readings = [(f, "", None, 0.0) for f in (5, 10, 15)]
    with pytest.raises(InterruptedError):
        supplement_reading(
            item,
            readings,
            0,
            20,
            lambda f: ("MD:88.1 TX:0.7", (88.1, 0.7), 0.99),
            cancelled=lambda: len(readings) > 3,
        )
    assert len(readings) == 4
