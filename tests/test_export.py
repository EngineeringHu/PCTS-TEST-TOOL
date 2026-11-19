from copy import deepcopy
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pytest

from pcts_video_tool.exporting import export_report
from pcts_video_tool.video import inspect_video


def test_xlsx_cache_matches_selected_policy_and_all_ten_rounds(session, generated_video, tmp_path):
    video, rois = generated_video
    session.video = inspect_video(str(video))
    session.rounds = [deepcopy(session.rounds[0]) for _ in range(10)]
    session.rounds[0].green.frame = 34
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    for mode in ("frames", "timestamps"):
        session.method = mode
        path = export_report(session, tmp_path / mode, evidence=False)
        detail = path.parent / "测量明细-草稿.xlsx"
        with ZipFile(detail) as archive:
            tree = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
            values = {n.attrib["r"]: n for n in tree.findall(".//m:c", ns)}
            assert float(values["H6"].find("m:v", ns).text) == pytest.approx(100, abs=0.5)
            assert float(values["H18"].find("m:v", ns).text) == pytest.approx(55, abs=0.5)
            formula = values["H6"].find("m:f", ns).text
            assert ("事件时间与来源" in formula) == (mode == "timestamps")
        assert (path.parent / "results.json").is_file()


def test_export_does_not_overwrite_existing_directory(session, generated_video, tmp_path):
    session.video = inspect_video(str(generated_video[0]))
    with pytest.raises(FileExistsError):
        export_report(session, tmp_path, evidence=False)


def template_location(book):
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    relationship = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    workbook = ET.fromstring(book.read("xl/workbook.xml"))
    sheet = next(s for s in workbook.find(ns + "sheets") if s.get("name") == "T1 (2)")
    relationships = ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
    location = next(
        r.get("Target") for r in relationships if r.get("Id") == sheet.get(relationship)
    )
    return location.lstrip("/") if location.startswith("/") else "xl/" + location


def template_cells(path):
    with ZipFile(path) as book:
        tree = ET.fromstring(book.read(template_location(book)))
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    return {c.get("r"): c for c in tree.iter(ns + "c")}


def cached(cells, address):
    node = cells[address].find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
    return None if node is None or node.text is None else float(node.text)


@pytest.mark.parametrize("mode", ["frames", "timestamps"])
def test_template_mapping_formulas_caches_and_original_parts_are_preserved(
    session, generated_video, tmp_path, mode
):
    from pcts_video_tool.models import EVENTS, calculate
    from pcts_video_tool.template_export import template_path

    original = template_path().read_bytes()
    session.video = inspect_video(str(generated_video[0]))
    session.method = mode
    session.rounds = [deepcopy(session.rounds[0]) for _ in range(10)]
    for i, item in enumerate(session.rounds):
        for event in EVENTS:
            getattr(item, event).frame += i * 70
    session.rounds[0].green.frame = 34
    target = export_report(session, tmp_path / "template", evidence=False)
    cells = template_cells(target)
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    for i, item in enumerate(session.rounds, 6):
        for col, value in zip(
            ("A", "B", "C", "G", "I", "J"),
            [getattr(item, k).frame for k in EVENTS] + [item.md, item.tx],
        ):
            assert cached(cells, f"{col}{i}") == value
        expected = calculate(session, item)
        for col, key in [
            ("O", "end_to_end"),
            ("P", "hu_input"),
            ("Q", "blink_interval"),
            ("R", "hu_output"),
            ("S", "md_total"),
            ("T", "flash_accuracy"),
            ("V", "hu_total"),
        ]:
            assert cached(cells, f"{col}{i}") == pytest.approx(expected[key])
    assert cached(cells, "O17") == pytest.approx(55)
    assert cached(cells, "O18") == pytest.approx(15.811388300841896)
    assert "O6:O15" in cells["O17"].find(ns + "f").text
    for row in (6, 7, 13):
        assert f"C{row}" in cells[f"F{row}"].find(ns + "f").text
        assert f"A{row}" in cells[f"M{row}"].find(ns + "f").text
    assert ("AB13" if mode == "timestamps" else "G13") in cells["O13"].find(ns + "f").text
    assert cached(cells, "E6") == session.rounds[0].blink1.frame
    assert cached(cells, "F7") == session.rounds[1].blink2.frame
    assert all(not c.find(ns + "f").attrib for c in cells.values() if c.find(ns + "f") is not None)
    assert cells["I17"].get("t") != "e"
    with ZipFile(target) as output, ZipFile(template_path()) as source:
        location = template_location(source)
        assert template_location(output) == location
        assert output.namelist() == source.namelist()
        for name in source.namelist():
            if name not in (location, "xl/workbook.xml"):
                assert output.read(name) == source.read(name)
        before = ET.fromstring(source.read(location))
        after = ET.fromstring(output.read(location))
        for index in range(1, 25):
            old = next(
                c
                for c in before.find(ns + "cols")
                if int(c.get("min")) <= index <= int(c.get("max"))
            )
            new = next(
                c
                for c in after.find(ns + "cols")
                if int(c.get("min")) <= index <= int(c.get("max"))
            )
            assert {k: v for k, v in old.attrib.items() if k not in ("min", "max")} == {
                k: v for k, v in new.attrib.items() if k not in ("min", "max")
            }
        spans = list(after.find(ns + "cols"))
        assert all(int(a.get("max")) < int(b.get("min")) for a, b in zip(spans, spans[1:]))
        assert next(c for c in spans if c.get("min") == "25").get("hidden") == "1"
        before_cells = {c.get("r"): c for c in before.iter(ns + "c")}
        assert all(c.get("s") == cells[a].get("s") for a, c in before_cells.items())
        book = ET.fromstring(output.read("xl/workbook.xml"))
        assert book.find(ns + "calcPr").get("fullCalcOnLoad") == "1"
        active = int(book.find(ns + "bookViews/" + ns + "workbookView").get("activeTab", "0"))
        assert list(book.find(ns + "sheets"))[active].get("name") == "T1 (2)"
    assert template_path().read_bytes() == original
    assert list(target.parent.glob("测量明细*.xlsx"))


def test_timestamp_export_keeps_frames_and_uses_pts_across_long_interval(
    session, generated_video, tmp_path
):
    session.video = inspect_video(str(generated_video[0]))
    item = session.rounds[0]
    for event, frame in zip(("press", "blink1", "blink2", "green"), (490, 494, 504, 510)):
        getattr(item, event).frame = frame
    session.method = "timestamps"
    path = export_report(session, tmp_path / "pts", evidence=False)
    cells = template_cells(path)
    assert cached(cells, "A6") == 490 and cached(cells, "G6") == 510
    assert cached(cells, "O6") == pytest.approx(22 * 1000 / 240)
    assert cached(cells, "Y6") == session.video.times[490]
    assert cached(cells, "AB6") == session.video.times[510]
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    assert "$B$20" not in cells["O6"].find(ns + "f").text
    session.video.times[510] = None
    path = export_report(session, tmp_path / "missing-pts", evidence=False)
    cells = template_cells(path)
    assert cached(cells, "G6") == 510 and cached(cells, "O6") is None
    assert cached(cells, "P6") == pytest.approx(4 * 1000 / 240)


@pytest.mark.parametrize("missing", ["press", "blink1", "green", "tx"])
def test_partial_inputs_do_not_turn_into_zero_or_erase_independent_metrics(
    session, generated_video, tmp_path, missing
):
    from pcts_video_tool.models import calculate

    session.video = inspect_video(str(generated_video[0]))
    item = session.rounds[0]
    if missing == "tx":
        item.tx = None
    else:
        getattr(item, missing).frame = None
    path = export_report(session, tmp_path / missing, evidence=False)
    cells = template_cells(path)
    metric = calculate(session, item)
    for col, key in [
        ("O", "end_to_end"),
        ("P", "hu_input"),
        ("Q", "blink_interval"),
        ("R", "hu_output"),
        ("S", "md_total"),
        ("T", "flash_accuracy"),
        ("V", "hu_total"),
    ]:
        assert cached(cells, col + "6") == metric[key]
        assert cached(cells, col + "17") == metric[key]
        assert cached(cells, col + "18") is None  # One observation is insufficient for sample SD.
    for row in range(7, 16):
        assert all(
            cached(cells, col + str(row)) is None
            for col in (
                "A",
                "B",
                "C",
                "E",
                "F",
                "G",
                "I",
                "J",
                "L",
                "M",
                "O",
                "P",
                "Q",
                "R",
                "S",
                "T",
                "U",
                "V",
            )
        )


def test_template_zero_values_are_not_missing_and_empty_report_has_no_stale_data(
    session, generated_video, tmp_path
):
    session.video = inspect_video(str(generated_video[0]))
    session.rounds[0].press.frame = 0
    session.rounds[0].md = session.rounds[0].tx = 0
    path = export_report(session, tmp_path / "zeros", evidence=False)
    cells = template_cells(path)
    assert cached(cells, "A6") == 0 and cached(cells, "S6") == 0
    session.rounds = []
    path = export_report(session, tmp_path / "empty", evidence=False)
    cells = template_cells(path)
    assert (
        cached(cells, "A6") is None
        and cached(cells, "O17") is None
        and cached(cells, "I18") is None
    )


def test_template_rejects_over_capacity_without_silently_dropping_rounds(
    session, generated_video, tmp_path
):
    session.video = inspect_video(str(generated_video[0]))
    session.rounds = [deepcopy(session.rounds[0]) for _ in range(11)]
    destination = tmp_path / "too-many"
    with pytest.raises(ValueError, match="11轮"):
        export_report(session, destination, evidence=False)
    assert not destination.exists()


def test_cancelled_export_never_publishes_a_final_template_report(
    session, generated_video, tmp_path
):
    session.video = inspect_video(str(generated_video[0]))
    cancel = {"value": False}

    def progress(message):
        if "填写Excel模板" in message:
            cancel["value"] = True

    folder = tmp_path / "cancelled"
    with pytest.raises(InterruptedError):
        export_report(
            session, folder, evidence=False, cancelled=lambda: cancel["value"], progress=progress
        )
    assert not list(folder.glob("测量结果*.xlsx"))
    assert not list(folder.glob("测量明细*.xlsx"))
