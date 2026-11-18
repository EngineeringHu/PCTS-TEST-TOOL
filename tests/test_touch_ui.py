import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QMessageBox, QSpinBox

from pcts_video_tool import ui
from pcts_video_tool.ui import CalibrationCanvas, MainWindow
from pcts_video_tool.video import inspect_video


def test_calibration_canvas_three_points_respect_letterboxing_and_undo():
    app = QApplication.instance() or QApplication([])
    canvas = CalibrationCanvas()
    canvas.resize(700, 500)
    canvas.set_image(np.zeros((120, 200, 3), np.uint8))
    canvas.show()
    app.processEvents()
    rect = canvas.image_rect()
    for x, y in ((0.4, 0.2), (0.4, 0.8), (0.6, 0.5)):
        QTest.mouseClick(
            canvas,
            Qt.LeftButton,
            pos=QPoint(
                round(rect.x() + x * (rect.width() - 1)), round(rect.y() + y * (rect.height() - 1))
            ),
        )
    assert np.asarray(canvas.points) == pytest.approx(
        np.array([[0.4, 0.2], [0.4, 0.8], [0.6, 0.5]]), abs=0.002
    )
    QTest.mouseClick(canvas, Qt.RightButton)
    assert len(canvas.points) == 2
    canvas.close()


def test_save_calibration_preserves_other_fields_and_roi_edit_clears_it(
    session, generated_video, monkeypatch
):
    app = QApplication.instance() or QApplication([])
    session.video = inspect_video(str(generated_video[0]))
    session.rois["touch"] = [0, 0, 1, 0, 1, 1, 0, 1]
    session.rounds[0].press.state = "review"
    window = MainWindow()
    window.accept_session(session)
    window.table.selectRow(0)
    window.show_frame(12)
    window.review_event("green")
    monkeypatch.setattr(ui, "ContactTracker", lambda *args: None)

    def set_points():
        dialog = app.activeModalWidget()
        dialog.findChild(QSpinBox).setValue(16)
        canvas = dialog.findChild(CalibrationCanvas)
        canvas.points = [[0.4, 0.2], [0.4, 0.8], [0.6, 0.5]]
        canvas.points_changed.emit()
        dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok).click()

    QTimer.singleShot(0, set_points)
    window.configure_touch()
    assert session.touch_calibration.frame == 16 and window.current_frame == 16
    assert window.comparison_field is None and window.video_views.currentWidget() is window.canvas
    assert session.touch_calibration.line == pytest.approx([0.4, 0.2, 0.4, 0.8])
    assert session.rounds[0].press.frame is None
    assert session.rounds[0].blink1.frame == 14 and session.rounds[0].md == 11.5
    assert window.calibration_button.text() == "触摸已校准"
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Yes)
    window.region_changed("touch", [0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9])
    assert session.touch_calibration is None
    assert window.calibration_button.text() == "触摸校准"
    window.dirty = False
    window.close()
