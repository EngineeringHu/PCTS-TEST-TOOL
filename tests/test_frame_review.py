import json
from copy import deepcopy
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from pcts_video_tool import analysis, ui
from pcts_video_tool.models import load_session, save_session
from pcts_video_tool.ui import MainWindow, Worker
from pcts_video_tool.video import crop, inspect_video, read_frame


@pytest.fixture
def review_window(session, generated_video):
    app = QApplication.instance() or QApplication([])
    session.video = inspect_video(str(generated_video[0]))
    session.rois = deepcopy(generated_video[1])
    session.rounds[0].green.frame = 50
    session.rounds[0].reading_frame = 65
    window = MainWindow()
    window.accept_session(session)
    window.show()
    app.processEvents()
    yield window
    window.dirty = False
    window.close()


def click_cell(window, row, column):
    rectangle = window.table.visualItemRect(window.table.item(row, column))
    QTest.mouseClick(window.table.viewport(), Qt.LeftButton, pos=rectangle.center())


@pytest.mark.parametrize(
    "saved_end,legacy_report,expected", [(83, False, 84), (None, True, 84), (None, False, 66)]
)
def test_add_tenth_round_uses_previous_end_and_expires_hint(
    review_window, tmp_path, saved_end, legacy_report, expected
):
    window = review_window
    session = window.session
    session.rounds = [deepcopy(session.rounds[0]) for _ in range(9)]
    session.rounds[-1].last_green_frame = saved_end
    path = tmp_path / "nine.session.json"
    save_session(session, path)
    if saved_end is None:
        raw = json.loads(path.read_text())
        raw["format"] = 5
        raw["diagnostic_dir"] = str(tmp_path)
        for row in raw["rounds"]:
            del row["last_green_frame"]
        path.write_text(json.dumps(raw))
        if legacy_report:
            (tmp_path / "detection.json").write_text(
                json.dumps({"round_candidates": [{"blink1": 14, "blink2": 20, "green_end": 84}]})
            )
    window.accept_session(load_session(path))
    click_cell(window, 0, 4)  # Adding from comparison view must still show the original video.
    window.add_round()
    assert window.table.currentRow() == 9 and window.frame_input.value() == expected
    assert window.video_views.currentWidget() is window.canvas and not window.canvas.pixmap.isNull()
    assert window.session.rounds[-1] == ui.Round()
    assert window.round_hint.isVisible() and "第10轮" in window.round_hint.text()
    assert window.round_hint_timer.isActive()
    window.round_hint_timer.start(1)
    QTest.qWait(40)
    assert not window.round_hint.isVisible()
    assert window.current_frame == expected


@pytest.mark.parametrize(
    "column,field,frame",
    [
        (1, "press", 10),
        (2, "blink1", 14),
        (3, "blink2", 20),
        (4, "green", 50),
        (5, "reading", 65),
        (6, "reading", 65),
    ],
)
def test_single_click_maps_field_to_original_frame_and_roi(review_window, column, field, frame):
    window = review_window
    click_cell(window, 0, column)
    assert window.comparison_field == field
    assert window.video_views.currentWidget() is window.comparison
    assert window.comparison.frame_indices == [frame - 1, frame, frame + 1]
    assert window.current_frame == frame and window.frame_input.value() == frame
    assert str(frame) in window.comparison.boxes[1].title()
    original = read_frame(window.session.video, frame)
    region = ui.REVIEW_REGIONS[field]
    expected = crop(original, window.session.rois[region])
    if region != "touch":
        assert window.comparison.panes[1].pixmap.width() == expected.shape[1]
        assert window.comparison.panes[1].pixmap.height() == expected.shape[0]
    assert not window.region_visibility.isEnabled()


def test_three_frames_show_actual_black_to_green_transition_and_browsing_does_not_edit(
    review_window,
):
    window = review_window
    click_cell(window, 0, 4)
    images = [pane.pixmap.toImage() for pane in window.comparison.panes]
    assert images[0] != images[1] and images[1] == images[2]
    window.step(1)
    assert window.comparison.frame_indices == [50, 51, 52]
    assert window.session.rounds[0].green.frame == 50
    window.select_current("green")
    assert window.session.rounds[0].green.frame == 51
    window.comparison.crop_only.setChecked(False)
    assert window.comparison.panes[1].pixmap.size() == window.canvas.pixmap.size()


def test_missing_candidates_and_video_boundaries_never_reuse_old_frames(review_window):
    window = review_window
    for frame, expected in ((0, [None, 0, 1]), (999, [998, 999, None])):
        window.session.rounds[0].green.frame = frame
        click_cell(window, 0, 4)
        assert window.comparison.frame_indices == expected
        for i, index in enumerate(expected):
            assert window.comparison.panes[i].pixmap.isNull() == (index is None)
    window.session.rounds[0].green.frame = None
    click_cell(window, 0, 4)
    assert window.current_frame == -1 and window.frame_input.value() == -1
    assert window.comparison.frame_indices == [None, None, None]
    assert all(pane.pixmap.isNull() for pane in window.comparison.panes)
    window.select_current("green")
    assert window.session.rounds[0].green.frame is None
    window.select_current("press")
    window.select_current("reading")
    assert window.session.rounds[0].press.frame == 10
    assert window.session.rounds[0].reading_frame == 65
    window.step(1)
    assert window.current_frame == 0 and window.comparison.frame_indices == [None, 0, 1]


def test_row_changes_roi_editing_and_task_replacement_keep_review_context_consistent(review_window):
    window = review_window
    second = deepcopy(window.session.rounds[0])
    second.green.frame, second.reading_frame = 150, 165
    window.session.rounds.append(second)
    window.refresh_table()
    click_cell(window, 0, 4)
    click_cell(window, 1, 5)
    assert window.comparison.frame_indices == [164, 165, 166]
    assert "第2轮" in window.comparison.title.text()
    window.region_combo.setCurrentIndex(2)
    assert window.comparison_field is None and window.video_views.currentWidget() is window.canvas
    assert window.canvas.pixmap.size().width() == window.session.video.width
    click_cell(window, 1, 4)
    replacement = deepcopy(window.session)
    replacement.rounds = []
    window.accept_session(replacement)
    assert window.comparison_field is None and window.video_views.currentWidget() is window.canvas


@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled"])
def test_analysis_clock_stops_on_every_outcome_and_restarts_without_import_reset(
    monkeypatch, outcome
):
    app = QApplication.instance() or QApplication([])
    now = {"value": 100.0}
    monkeypatch.setattr(ui, "perf_counter", lambda: now["value"])
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: None)

    def synchronous_start(worker):
        worker.run()
        worker.finished.emit()

    monkeypatch.setattr(Worker, "start", synchronous_start)
    window = MainWindow()

    def work(progress, cancelled, percentage):
        assert window.analysis_clock_timer.isActive()
        assert window.elapsed_label.text().endswith("00:00:00")
        now["value"] += 4.25
        percentage(42)
        if outcome == "failure":
            raise ValueError("测试失败")
        if outcome == "cancelled":
            raise InterruptedError("已取消")

    window.begin(work, lambda result: None, measure_analysis=True)
    assert not window.analysis_clock_timer.isActive()
    assert window.analysis_elapsed_seconds == 4.25
    assert window.elapsed_label.text().endswith("00:00:04")
    now["value"] += 50
    app.processEvents()
    window.update_analysis_clock()
    assert window.elapsed_label.text().endswith("00:00:04")
    window.begin(lambda *args: None, lambda result: None)
    assert window.analysis_elapsed_seconds == 4.25
    window.begin(work, lambda result: None, measure_analysis=True)
    assert window.analysis_elapsed_seconds == 4.25
    window.close()


@pytest.mark.parametrize(
    "error,status", [(ValueError("failed"), "failed"), (InterruptedError("cancel"), "cancelled")]
)
def test_analysis_diagnostic_records_elapsed_time_on_failure_and_cancel(
    session, monkeypatch, error, status
):
    clock = iter([10.0, 12.5])
    monkeypatch.setattr(analysis, "perf_counter", lambda: next(clock))

    def fail(*args):
        raise error

    monkeypatch.setattr(analysis, "_analyze", fail)
    with pytest.raises(type(error)):
        analysis.analyze(session)
    record = json.loads((Path(session.diagnostic_dir) / "status.json").read_text())
    assert record["status"] == status and record["elapsed_seconds"] == 2.5


@pytest.mark.parametrize("column", [0, 7, 8])
def test_round_summary_cells_also_open_default_three_frame_review(review_window, column):
    window = review_window
    click_cell(window, 0, column)
    assert window.video_views.currentWidget() is window.comparison
    assert window.comparison.frame_indices == [9, 10, 11]


def test_single_full_frame_choice_persists_across_cells_and_rows(review_window):
    window = review_window
    second = deepcopy(window.session.rounds[0])
    second.blink1.frame = 114
    window.session.rounds.append(second)
    window.refresh_table()
    click_cell(window, 0, 4)
    window.comparison.back_button.click()
    assert window.video_views.currentWidget() is window.canvas
    click_cell(window, 0, 2)
    assert window.video_views.currentWidget() is window.canvas
    assert window.current_frame == 14
    click_cell(window, 1, 2)
    assert window.video_views.currentWidget() is window.canvas
    assert window.current_frame == 114
    window.step(2)
    window.review_layout.setCurrentIndex(0)
    assert window.video_views.currentWidget() is window.comparison
    assert window.comparison.frame_indices == [115, 116, 117]
    click_cell(window, 1, 4)
    assert window.video_views.currentWidget() is window.comparison
    assert window.comparison.frame_indices == [49, 50, 51]


def test_uncropped_three_frames_remain_uncropped_when_changing_fields(review_window):
    window = review_window
    click_cell(window, 0, 4)
    window.comparison.crop_only.setChecked(False)
    click_cell(window, 0, 2)
    assert window.video_views.currentWidget() is window.comparison
    assert not window.comparison.crop_only.isChecked()
    assert window.comparison.frame_indices == [13, 14, 15]
    assert all(
        pane.pixmap.width() == window.session.video.width for pane in window.comparison.panes
    )


@pytest.mark.parametrize("full_frame", [False, True])
def test_missing_press_opens_blink1_minus_twenty_without_writing_a_measurement(
    review_window, full_frame
):
    window = review_window
    item = window.session.rounds[0]
    item.press.frame, item.blink1.frame = None, 80
    window.refresh_table()
    if full_frame:
        window.review_layout.setCurrentIndex(1)
    window.dirty = False
    click_cell(window, 0, 1)
    assert window.current_frame == 60 and window.frame_input.value() == 60
    assert item.press.frame is None and window.event_inputs["press"].value() == -1
    assert window.table.item(0, 1).text() == "—" and not window.dirty
    assert "60" in window.review_hint.text() and "未写入press" in window.review_hint.text()
    assert window.video_views.currentWidget() is (
        window.canvas if full_frame else window.comparison
    )
    if not full_frame:
        assert window.comparison.frame_indices == [59, 60, 61]
        assert (
            window.comparison.panes[1].pixmap.width() == window.session.video.width
        )  # Touch ROI, not blink ROI.
    window.step(2)
    assert item.press.frame is None and not window.dirty
    window.select_current("press")
    assert item.press.frame == 62
    assert not window.review_hint.isVisible()


@pytest.mark.parametrize("blink1,expected", [(14, 0), (20, 0), (21, 1), (None, -1)])
def test_missing_press_search_start_respects_video_start_and_missing_blink1(
    review_window, blink1, expected
):
    window = review_window
    click_cell(window, 0, 4)
    item = window.session.rounds[0]
    item.press.frame, item.blink1.frame = None, blink1
    window.refresh_table()
    click_cell(window, 0, 1)
    assert window.current_frame == expected and item.press.frame is None
    if expected < 0:
        assert window.comparison.frame_indices == [None, None, None]
        assert "blink 1" in window.review_hint.text()
    else:
        assert window.comparison.frame_indices == [
            None if expected == 0 else expected - 1,
            expected,
            expected + 1,
        ]


def test_missing_press_uses_the_clicked_rounds_blink1(review_window):
    window = review_window
    window.session.rounds[0].press.frame = None
    window.session.rounds[0].blink1.frame = 80
    second = deepcopy(window.session.rounds[0])
    second.blink1.frame = 180
    window.session.rounds.append(second)
    window.refresh_table()
    click_cell(window, 0, 1)
    assert window.current_frame == 60
    click_cell(window, 1, 1)
    assert window.current_frame == 160
    assert window.comparison.frame_indices == [159, 160, 161]
    assert all(item.press.frame is None for item in window.session.rounds)


def test_rapid_frame_steps_and_cached_revisit_preserve_measurements(review_window):
    window = review_window
    original = deepcopy(window.session.rounds)
    click_cell(window, 0, 4)
    seen = []
    for step in [1] * 12 + [-1] * 20 + [1] * 8:
        previous = window.current_frame
        window.step(step)
        assert window.current_frame == previous + step
        assert window.comparison.frame_indices == [
            window.current_frame - 1,
            window.current_frame,
            window.current_frame + 1,
        ]
        seen.append(window.current_frame)
    assert seen[-1] == 50 and window.session.rounds == original
    window.step(4)
    click_cell(window, 0, 4)
    assert window.current_frame == 50  # Clicking the same cell still returns to the event frame.


def test_reader_is_released_on_task_replacement_and_window_close(review_window, generated_video):
    window = review_window
    click_cell(window, 0, 4)
    old = window.frame_reader
    assert old.container is not None and old.cache
    replacement = deepcopy(window.session)
    replacement.rounds = []
    window.accept_session(replacement)
    assert old.container is None and old.decoder is None and not old.cache
    assert window.frame_reader is not old
    active = window.frame_reader
    window.dirty = False
    window.close()
    assert active.container is None and not active.cache


def test_neighbor_failure_stays_empty_and_center_failure_clears_old_picture(
    review_window, monkeypatch
):
    window = review_window
    actual = window.frame_reader.read
    broken = {49}

    def reading(index):
        if index in broken:
            raise ValueError("模拟读取失败")
        return actual(index)

    monkeypatch.setattr(window.frame_reader, "read", reading)
    click_cell(window, 0, 4)
    assert window.current_frame == 50
    assert window.comparison.panes[0].pixmap.isNull()
    assert not window.comparison.panes[1].pixmap.isNull()
    errors = []
    monkeypatch.setattr(window, "error", errors.append)
    broken.add(50)
    click_cell(window, 0, 4)
    assert window.current_frame == -1 and window.frame_input.value() == -1
    assert all(pane.pixmap.isNull() for pane in window.comparison.panes)
    assert window.session.rounds[0].green.frame == 50 and errors
