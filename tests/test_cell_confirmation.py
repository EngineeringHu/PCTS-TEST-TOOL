import json
from copy import deepcopy

import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit, QMenu

from pcts_video_tool.models import Event, load_session, save_session
from pcts_video_tool.ui import CELL_COLORS, MainWindow
from pcts_video_tool.video import inspect_video


@pytest.fixture
def window(session, generated_video):
    app = QApplication.instance() or QApplication([])
    session.video = inspect_video(str(generated_video[0]))
    item = session.rounds[0]
    item.press = Event(10, "review", "候选")
    item.blink1 = Event(14, "auto", "自动通过")
    item.blink2 = Event(None)
    item.green = Event(50, "auto", "自动通过")
    item.reading_state = "review"
    window = MainWindow()
    window.accept_analysis(session)
    window.show()
    app.processEvents()
    yield window
    window.dirty = False
    window.close()


def color(window, row, column):
    return window.table.item(row, column).background().color().name()


def menu_at(window, row, column, monkeypatch, action=1):
    position = window.table.visualItemRect(window.table.item(row, column)).center()
    seen = []

    def choose():
        menu = QApplication.activePopupWidget()
        assert isinstance(menu, QMenu)
        actions = menu.actions()
        seen.extend((item.text(), item.isEnabled()) for item in actions)
        if actions[action].isEnabled():
            menu.setActiveAction(actions[action])
            QTest.keyClick(menu, Qt.Key_Return)
        else:
            QTest.keyClick(menu, Qt.Key_Escape)

    # Right press must not invoke selection navigation before the menu opens.
    QTest.mousePress(window.table.viewport(), Qt.RightButton, pos=position)
    QTest.mouseRelease(window.table.viewport(), Qt.RightButton, pos=position)
    QTimer.singleShot(0, choose)
    window.cell_menu(position)
    return seen


def test_analysis_colors_are_independent_per_cell(window):
    assert color(window, 0, 1) == CELL_COLORS["review"]
    assert color(window, 0, 2) == CELL_COLORS["accepted"]
    assert color(window, 0, 3) == CELL_COLORS["missing"]
    assert color(window, 0, 4) == CELL_COLORS["accepted"]
    assert color(window, 0, 5) == CELL_COLORS["review"]
    assert "自动通过" in window.table.item(0, 2).toolTip()


@pytest.mark.parametrize("column", [1, 2, 3, 4])
def test_right_click_confirms_displayed_frame_even_on_another_row(window, monkeypatch, column):
    first = deepcopy(window.session.rounds[0])
    window.session.rounds.append(deepcopy(first))
    window.refresh_table()
    window.table.setCurrentCell(0, 4)
    window.review_cell(0, 4)
    window.show_frame(100)
    actions = menu_at(window, 1, column, monkeypatch)
    assert "100" in actions[1][0]
    event = getattr(window.session.rounds[1], ("press", "blink1", "blink2", "green")[column - 1])
    assert event.frame == 100 and event.state == "manual"
    assert window.current_frame == window.frame_input.value() == 100
    assert window.session.rounds[0] == first
    assert color(window, 1, column) == CELL_COLORS["accepted"]


def test_candidate_menu_only_browses_and_missing_frame_cannot_confirm(window, monkeypatch):
    window.show_frame(100)
    original = deepcopy(window.session.rounds)
    window.dirty = False
    menu_at(window, 0, 1, monkeypatch, action=0)
    assert window.current_frame == 10 and window.session.rounds == original and not window.dirty
    window.show_frame(-1)
    actions = menu_at(window, 0, 1, monkeypatch)
    assert not actions[1][1]
    assert window.session.rounds == original


def test_readings_edit_and_confirm_independently_and_persist(window, monkeypatch, tmp_path):
    item = window.session.rounds[0]
    item.reading_state = "auto"
    window.refresh_table()
    table = window.table
    pos = table.visualItemRect(table.item(0, 5)).center()
    QTest.mouseClick(table.viewport(), Qt.LeftButton, pos=pos)
    QTest.mouseDClick(table.viewport(), Qt.LeftButton, pos=pos)
    editor = table.findChild(QLineEdit)
    assert editor is not None and editor.isVisible()
    editor.selectAll()
    QTest.keyClicks(editor, "-12.5")
    QTest.keyClick(editor, Qt.Key_Return)
    QApplication.processEvents()
    assert item.md == -12.5 and window.md.text() == "-12.5"
    assert item.reading_state == "review"
    assert color(window, 0, 5) == CELL_COLORS["review"]
    assert color(window, 0, 6) == CELL_COLORS["accepted"]
    window.show_frame(100)
    menu_at(window, 0, 5, monkeypatch, action=0)
    assert item.reading_states == {"md": "manual", "tx": "auto"}
    assert window.current_frame == 100
    assert color(window, 0, 5) == CELL_COLORS["accepted"]
    window.tx.setText("8.2")
    window.tx.textEdited.emit("8.2")
    assert color(window, 0, 5) == CELL_COLORS["accepted"]
    assert color(window, 0, 6) == CELL_COLORS["review"]
    path = tmp_path / "confirmed.json"
    save_session(window.session, path)
    restored = load_session(path)
    assert restored.rounds[0].reading_states == {"md": "manual", "tx": "review"}
    raw = json.loads(path.read_text())
    raw["format"] = 6
    raw["rounds"][0].pop("reading_states")
    raw["rounds"][0]["reading_state"] = "manual"
    path.write_text(json.dumps(raw))
    window.accept_session(load_session(path))
    assert color(window, 0, 5) == color(window, 0, 6) == CELL_COLORS["accepted"]


def test_invalid_table_number_preserves_previous_value_and_confirmation(window):
    item = window.session.rounds[0]
    before = deepcopy(item)
    window.dirty = False
    window.table.item(0, 5).setText("nan")
    assert item == before and not window.dirty
    assert window.table.item(0, 5).text() == str(item.md)
    assert "有限数字" in window.statusBar().currentMessage()


def test_toggling_hints_does_not_change_confirmed_readings(window):
    window.confirm_cell(0, 5, window.current_frame)
    before = deepcopy(window.session.rounds)
    window.dirty = False
    window.missing_hints_button.setChecked(False)
    window.missing_hints_button.setChecked(True)
    assert window.session.rounds == before and not window.dirty
    assert color(window, 0, 5) == CELL_COLORS["accepted"]
