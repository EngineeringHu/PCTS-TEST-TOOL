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
        with ZipFile(path) as archive:
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
