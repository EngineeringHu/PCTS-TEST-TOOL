import os
from copy import deepcopy

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QMessageBox, QSpinBox

from pcts_video_tool.ui import MainWindow, Worker
from pcts_video_tool.video import inspect_video


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


def test_replacing_selected_task_refreshes_editor(application, session, generated_video):
    session.video = inspect_video(str(generated_video[0]))
    window = MainWindow()
    window.accept_session(session)
    window.table.selectRow(0)
    assert window.event_inputs["press"].value() == 10
    second = deepcopy(session)
    for name in ("press", "blink1", "blink2", "green"):
        getattr(second.rounds[0], name).frame += 100
    window.accept_session(second)
    assert window.event_inputs["press"].value() == 110
    window.confirm_round()
    assert window.session.rounds[0].press.frame == 110
    window.dirty = False
    window.close()


def test_edits_are_preserved_before_confirmation(application, session, generated_video):
    session.video = inspect_video(str(generated_video[0]))
    session.rounds.append(deepcopy(session.rounds[0]))
    window = MainWindow()
    window.accept_session(session)
    window.table.selectRow(0)
    window.md.setText("999")
    window.md.textEdited.emit("999")
    window.event_inputs["press"].setValue(9)
    window.table.selectRow(1)
    window.table.selectRow(0)
    assert window.md.text() == "999.0"
    assert window.event_inputs["press"].value() == 9
    assert session.rounds[0].reading_state == "review"
    assert window.dirty
    window.method.setCurrentIndex(1)
    assert session.method == "timestamps"
    assert session.rounds[0].md == 999
    window.dirty = False
    window.close()


def test_empty_task_clears_previous_editor(application, session, generated_video):
    session.video = inspect_video(str(generated_video[0]))
    window = MainWindow()
    window.accept_session(session)
    window.table.selectRow(0)
    empty = deepcopy(session)
    empty.rounds = []
    window.accept_session(empty)
    assert window.event_inputs["press"].value() == -1
    assert window.md.text() == ""
    window.dirty = False
    window.close()


def test_manual_range_change_clears_stale_results(
    application, session, generated_video, monkeypatch
):
    session.video = inspect_video(str(generated_video[0]))
    window = MainWindow()
    window.accept_session(session)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Yes)

    def edit():
        dialog = application.activeModalWidget()
        assert isinstance(dialog, QDialog)
        dialog.findChild(QComboBox).setCurrentIndex(1)
        start, end = dialog.findChildren(QSpinBox)
        start.setValue(100)
        end.setValue(800)
        dialog.accept()

    QTimer.singleShot(0, edit)
    window.configure_range()
    assert (session.range_mode, session.test_start, session.test_end) == ("manual", 100, 800)
    assert session.range_confirmed and not session.rounds and window.dirty
    window.dirty = False
    window.close()


def test_no_results_shows_reason_instead_of_generic_ready(
    application, session, generated_video, monkeypatch
):
    session.video = inspect_video(str(generated_video[0]))
    session.rounds = []
    session.analysis_notes = ["未找到明确变绿事件"]
    notices = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: notices.append(args[2]))
    window = MainWindow()
    window.accept_analysis(session)
    assert notices and "未识别到有效轮次" in window.statusBar().currentMessage()
    assert "未找到明确变绿事件" in window.details.toPlainText()
    window.dirty = False
    window.close()


def test_analysis_selects_first_pending_round_and_explains_next_step(
    application, session, generated_video
):
    session.video = inspect_video(str(generated_video[0]))
    session.rounds.append(deepcopy(session.rounds[0]))
    session.rounds[1].press.frame = None
    window = MainWindow()
    window.accept_analysis(session)
    assert window.table.currentRow() == 1
    assert "分析完成" in window.statusBar().currentMessage()
    assert "复核" in window.statusBar().currentMessage()
    window.dirty = False
    window.close()


@pytest.mark.parametrize("outcome", ["success", "failed", "cancelled"])
def test_worker_progress_finishes_only_on_success(application, monkeypatch, outcome):
    window = MainWindow()
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: None)
    percentages = []

    def work(progress, cancelled, percentage):
        percentage(42)
        if outcome == "failed":
            raise ValueError("读取失败")
        if outcome == "cancelled":
            raise InterruptedError("已取消")
        return "done"

    worker = Worker(work)
    loop = QEventLoop()
    worker.percentage.connect(percentages.append)
    worker.percentage.connect(window.show_progress)
    worker.failed.connect(window.error)
    worker.finished.connect(loop.quit)
    QTimer.singleShot(5000, loop.quit)
    worker.start()
    loop.exec()
    assert not worker.isRunning()
    assert percentages == ([42, 100] if outcome == "success" else [42])
    assert window.progress_bar.value() == percentages[-1]
    if outcome == "cancelled":
        assert "已取消" in window.progress_bar.format()
    if outcome == "failed":
        assert "未完成" in window.progress_bar.format()
    window.close()
