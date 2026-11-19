from copy import deepcopy

import pytest
from PySide6.QtCore import QPropertyAnimation, QTimer, QUrl
from PySide6.QtGui import QTextDocument
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QHBoxLayout, QLabel, QTabWidget

from pcts_video_tool.models import Round
from pcts_video_tool.ui import CELL_COLORS, MainWindow
from pcts_video_tool.workflow_guide import EXAMPLES, REGIONS, DetectionGuide, guide_stage


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_dialog_follows_progress_and_keeps_diagnostics(app, session):
    def current(task):
        dialog = DetectionGuide(task, CELL_COLORS)
        tabs = dialog.findChild(QTabWidget)
        text = tabs.currentWidget().toPlainText()
        assert "诊断日志位置" in tabs.widget(tabs.count() - 1).toPlainText()
        dialog.close()
        return text

    introductory = current(None)
    assert all(
        word in introductory for word in ("导入原始视频", "固定拍摄机位", "每一轮都点击同一位置")
    )
    session.rounds.clear()
    session.rois = {"flash": [0, 0, 1, 0, 1, 1, 0, 1]}
    dialog = DetectionGuide(session, CELL_COLORS)
    page = dialog.findChild(QTabWidget).currentWidget()
    assert "已完成 1/4" in page.toPlainText()
    assert "1941" in page.toPlainText() and "橙圈" in page.toPlainText()
    assert page.toPlainText().count("待框选") == 3
    for name in REGIONS:
        url = QUrl.fromLocalFile(str(EXAMPLES / f"{name}.png"))
        assert url.toString() in page.toHtml()
        assert page.document().resource(QTextDocument.ImageResource, url).width() == 560
    dialog.close()
    session.rois = dict.fromkeys(REGIONS, session.rois["flash"])
    assert "开始自动分析" in current(session)
    session.analysis_notes = ["本轮未匹配到灭灯。"]
    session.video.warnings = ["部分帧缺少时间戳。"]
    assert "没有可复核的轮次" in current(session)
    session.rounds = [Round()]
    text = current(session)
    assert all(word in text for word in ("红", "黄", "绿", "右键", "双击", "确认当前帧"))
    dialog = DetectionGuide(session, CELL_COLORS)
    notes = dialog.findChild(QTabWidget).widget(2).toPlainText()
    assert session.analysis_notes[0] in notes and session.video.warnings[0] in notes
    dialog.close()


def test_guide_button_reminds_by_stage_and_runs_only_explicit_action(app, session, monkeypatch):
    w = MainWindow()
    title = next(label for label in w.findChildren(QLabel) if "VIDEO MEASUREMENT" in label.text())
    toolbar = next(layout for layout in w.findChildren(QHBoxLayout) if layout.indexOf(title) >= 0)
    assert toolbar.itemAt(toolbar.indexOf(title) + 1).widget() is w.guide_button
    assert w.guide_button.text() == "操作指引"
    monkeypatch.setattr(w, "show_frame", lambda *args, **kwargs: None)
    actions = []
    monkeypatch.setattr(w, "open_video", lambda: actions.append("import"))
    monkeypatch.setattr(w, "start_analysis", lambda: actions.append("analyze"))

    def open_guide(role):
        def press():
            dlg = app.activeModalWidget()
            box = dlg.findChild(QDialogButtonBox)
            next(b for b in box.buttons() if box.buttonRole(b) == role).click()

        QTimer.singleShot(0, press)
        w.guide_button.click()

    assert w.guide_pulse.state() == QPropertyAnimation.Running and w.guide_pulse.duration() >= 3000
    open_guide(QDialogButtonBox.RejectRole)
    assert not actions and w.guide_pulse.state() == QPropertyAnimation.Stopped
    open_guide(QDialogButtonBox.AcceptRole)
    assert actions == ["import"]
    session.rounds.clear()
    session.rois.clear()
    w.accept_session(session, saved=True)
    assert w.guide_pulse.state() == QPropertyAnimation.Running
    open_guide(QDialogButtonBox.AcceptRole)
    assert w.region_combo.currentData() == "flash" and not w.session.rois
    for key in REGIONS:
        w.region_changed(key, [0, 0, 1, 0, 1, 1, 0, 1])
    assert guide_stage(w.session) == "ready" and w.guide_pulse.state() == QPropertyAnimation.Running
    open_guide(QDialogButtonBox.RejectRole)
    assert actions == ["import"]
    open_guide(QDialogButtonBox.AcceptRole)
    assert actions == ["import", "analyze"]
    w.session.rounds = [Round()]
    w.refresh_table()
    assert w.guide_pulse.state() == QPropertyAnimation.Running
    w.show()
    app.processEvents()
    assert w.frame_input.lineEdit().font().pixelSize() >= 22
    before = deepcopy(w.session)
    open_guide(QDialogButtonBox.RejectRole)
    assert w.session == before and w.guide_pulse.state() == QPropertyAnimation.Stopped
    w.dirty = False
    w.close()


def test_worker_pauses_reminder_without_rearming_a_read_stage(app):
    from PySide6.QtCore import QEventLoop

    w = MainWindow()
    for already_read in (False, True):
        if already_read:
            QTimer.singleShot(0, lambda: app.activeModalWidget().reject())
            w.guide_button.click()
        finished = QEventLoop()
        w.begin(lambda *args: None, lambda result: None)
        assert w.guide_pulse.state() != QPropertyAnimation.Running
        w.worker.finished.connect(finished.quit)
        QTimer.singleShot(3000, finished.quit)
        finished.exec()
        assert w.root.isEnabled()
        expected = QPropertyAnimation.Stopped if already_read else QPropertyAnimation.Running
        assert w.guide_pulse.state() == expected
    w.close()
