import json
from copy import deepcopy
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from pcts_video_tool import analysis, ui
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
