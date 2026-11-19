from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QFileDialog

from pcts_video_tool import region_frames
from pcts_video_tool.models import save_session
from pcts_video_tool.ui import MainWindow, Worker
from pcts_video_tool.video import read_frame


@pytest.mark.parametrize(
    "texts,has_marker,has_reading",
    [
        (["PRESS"], False, False),
        (["PRESS", "HERE", "TX2"], True, True),
        (["PRESS:HERE", "MD"], True, True),
        (["PRESSHERE", "MD:62.7TX2"], True, True),
        (["HDMI", "CMD"], False, False),
    ],
)
def test_ocr_labels_for_selection(texts, has_marker, has_reading):
    box = np.float32([[0, 0], [20, 0], [20, 8], [0, 8]])
    result = SimpleNamespace(txts=texts, boxes=np.array([box] * len(texts)))
    marker, reading = region_frames._text_targets(result)
    assert (marker is not None, reading) == (has_marker, has_reading)


def test_import_reselect_and_reopen_use_original_images(generated_video, tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    path = str(generated_video[0])
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (path, ""))

    def start(worker):
        worker.run()
        worker.finished.emit()

    monkeypatch.setattr(Worker, "start", start)
    window = MainWindow()
    window.open_video()
    frames = {name: index for name, (index, _) in window.region_previews.items()}
    assert set(frames) == {"flash", "green", "reading"}  # Original contains text and no hands.
    assert 53 <= frames["reading"] % 100 < 80
    assert np.array_equal(
        window.region_previews["green"][1], read_frame(window.session.video, frames["green"])
    )
    window.dirty = False
    task = tmp_path / "with-previews.json"
    save_session(window.session, task)
    window.show_frame(12)
    with monkeypatch.context() as check:
        check.setattr(
            window.frame_reader, "read", lambda _: pytest.fail("selection decoded a frame")
        )
        for name in ("flash", "reading", "green", "flash"):
            window.region_combo.setCurrentIndex(window.region_combo.findData(name))
            assert (
                window.current_frame
                == window.frame_input.value()
                == window.slider.value()
                == frames[name]
            )
            assert window.canvas.shape == ("rectangle" if name == "flash" else "quad")
        window.shape_combo.setCurrentIndex(window.shape_combo.findData("quad"))
        assert window.canvas.shape == "quad"
    window.show_frame(12)
    window.region_combo.activated.emit(window.region_combo.currentIndex())
    assert window.current_frame == frames["flash"] and window.canvas.shape == "rectangle"
    window.show_frame(12)
    window.region_combo.setCurrentIndex(window.region_combo.findData("touch"))
    assert window.current_frame == 12 and window.canvas.shape == "quad"
    assert "手动" in window.statusBar().currentMessage()
    assert not window.dirty and not window.session.rois and not window.session.rounds
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(task), ""))
    monkeypatch.setattr(
        region_frames, "inspect_video", lambda *args, **kwargs: pytest.fail("reindexed saved video")
    )
    window.open_task()
    assert window.region_combo.currentIndex() == 0 and not window.dirty
    assert set(window.region_previews) == set(frames)
    window.close()
    app.processEvents()


def test_optional_model_failure_preserves_manual_import(generated_video, monkeypatch):
    def missing_model():
        raise FileNotFoundError("test model")

    monkeypatch.setattr(region_frames, "create_ocr", missing_model)
    session, previews = region_frames.prepare_region_session(
        str(generated_video[0]), lambda message: None, lambda: False
    )
    assert list(previews) == ["flash"] and previews["flash"][0] == 0
    assert np.array_equal(previews["flash"][1], read_frame(session.video, 0))
