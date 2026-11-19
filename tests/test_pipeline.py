import csv
import json
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from pcts_video_tool.analysis import analyze
from pcts_video_tool.exporting import export_report
from pcts_video_tool.models import Session, TouchCalibration, normalize_roi
from pcts_video_tool.run_logs import export_diagnostics
from pcts_video_tool.video import inspect_video


def delayed_first_green(detector, *, conflict=False):
    """Model a weak coarse score while keeping real decoded pixels for refinement."""

    def detect(*args):
        result = detector(*args)
        result.rounds[0].green.frame = 60
        result.rounds[0].green.state = "review"
        result.diagnostics["green_evidence"][0].update(first_green_candidate=60, confirmed_at=65)
        result.diagnostics["green_segments"][0] = (60, result.green_ends[0])
        result.diagnostics["round_candidates"][0]["green"] = 60
        if conflict:
            result.rounds[0].blink2.frame = 55
            result.diagnostics["round_candidates"][0]["blink2"] = 55
        return result

    return detect


def test_full_pipeline_uses_local_models_and_exports_evidence(
    generated_video, tmp_path, monkeypatch
):
    from zipfile import ZipFile

    from pcts_video_tool import analysis

    path, rois = generated_video
    original_read = analysis.read_numbers
    calls = 0

    def one_bad_seed(engine, image):
        nonlocal calls
        calls += 1
        if calls == 1:
            return "MD88.12 TX:-0.7", None, 0.99
        return original_read(engine, image)

    monkeypatch.setattr(analysis, "read_numbers", one_bad_seed)
    monkeypatch.setattr(analysis, "detect_rounds", delayed_first_green(analysis.detect_rounds))
    percentages = []
    result = analyze(Session(inspect_video(str(path)), rois), percentage=percentages.append)
    assert percentages[0] == 0 and percentages[-1] == 100
    assert percentages == sorted(percentages) and all(0 <= p <= 100 for p in percentages)
    assert len(result.rounds) == 10
    assert result.diagnostic_dir
    status = json.loads((Path(result.diagnostic_dir) / "status.json").read_text())
    assert status["status"] == "complete" and status["elapsed_seconds"] >= 0
    assert (Path(result.diagnostic_dir) / "signals.csv").is_file()
    assert (Path(result.diagnostic_dir) / "detection.json").is_file()
    ocr_checks = [
        json.loads(line)
        for line in (Path(result.diagnostic_dir) / "ocr-checks.jsonl").read_text().splitlines()
    ]
    assert len(ocr_checks) == 32
    assert sum(row["format_valid"] for row in ocr_checks) == 31
    assert [row["frame"] for row in ocr_checks if row["kind"] == "neighbor"] == [66, 68]
    assert [row["frame"] for row in ocr_checks if row["round"] == 1 and row["kind"] == "seed"] == [
        67,
        73,
        78,
    ]
    bundle = tmp_path / "diagnostics.zip"
    export_diagnostics(result, bundle)
    with ZipFile(bundle) as archive:
        assert "analysis/ocr-checks.jsonl" in archive.namelist()
        assert "analysis/green-refinement.jsonl" in archive.namelist()
    recorded = json.loads(
        (Path(result.diagnostic_dir) / "detection.json").read_text(encoding="utf-8")
    )
    assert len(recorded["round_candidates"]) == 10
    assert recorded["round_candidates"][0]["blink1"] == 20
    assert recorded["round_candidates"][0]["coarse_green"] == 60
    assert recorded["round_candidates"][0]["green"] == 50
    assert recorded["coarse_green_segments"][0][0] == 60
    assert recorded["green_segments"][0][0] == 50
    assert recorded["green_refinements"][0]["confirmed_at"] == 55
    with (Path(result.diagnostic_dir) / "signals.csv").open(newline="") as stream:
        signals = list(csv.DictReader(stream))
    assert len(signals) == 1000 and signals[-1]["frame"] == "999"
    assert float(signals[50]["green_ratio"]) > 0
    for i, item in enumerate(result.rounds):
        assert item.last_green_frame == i * 100 + 79
        assert (item.blink1.frame, item.blink2.frame, item.green.frame) == (
            i * 100 + 20,
            i * 100 + 40,
            i * 100 + 50,
        )
        assert (item.md, item.tx, item.reading_state) == (88.1, -0.7, "auto")
        assert item.press.frame is None  # The synthetic video intentionally has no hand.
        assert item.press.state == "review"
    report = export_report(result, tmp_path / "report")
    assert "草稿" in report.name
    assert len(list((report.parent / "evidence").rglob("*.png"))) == 40


def test_green_refinement_retains_coarse_candidate_when_event_order_conflicts(
    generated_video, monkeypatch
):
    from pcts_video_tool import analysis

    path, rois = generated_video
    monkeypatch.setattr(
        analysis, "detect_rounds", delayed_first_green(analysis.detect_rounds, conflict=True)
    )
    monkeypatch.setattr(analysis, "create_ocr", lambda: None)
    monkeypatch.setattr(analysis, "create_hand_detector", lambda **kwargs: None)
    session = Session(
        inspect_video(str(path)),
        rois,
        range_mode="manual",
        test_start=0,
        test_end=999,
        range_confirmed=True,
    )
    result = analyze(session)
    assert result.rounds[0].green.frame == 60 and result.rounds[0].blink2.frame == 55
    assert "顺序冲突" in result.rounds[0].green.reason
    record = json.loads(
        (Path(result.diagnostic_dir) / "detection.json").read_text(encoding="utf-8")
    )["green_refinements"][0]
    assert (
        record["candidate_frame"] == 50 and record["selected_frame"] == 60 and not record["applied"]
    )


def test_real_ocr_marker_excludes_intro_and_invalid_tail(generated_video, tmp_path):
    original, rois = generated_video
    path = tmp_path / "bounded-test.nut"
    with av.open(str(path), "w") as output, av.open(str(original)) as source:
        stream = output.add_stream("ffv1", rate=240)
        stream.width, stream.height, stream.pix_fmt = 640, 360, "bgr0"
        index = 0

        def write(image):
            nonlocal index
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts, frame.time_base = index, Fraction(1, 240)
            index += 1
            for packet in stream.encode(frame):
                output.mux(packet)

        intro = np.full((360, 640, 3), 230, np.uint8)
        cv2.putText(intro, "MENU", (270, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        for _ in range(120):
            write(intro)
        for frame in source.decode(video=0):
            write(frame.to_ndarray(format="bgr24"))
        for i in range(240):
            tail = np.full((360, 640, 3), (0, 170, 0), np.uint8)
            # The test display is blank in the middle while unrelated content remains outside it.
            cv2.putText(tail, "HOME", (45, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            tail[9:81, 16:96] = 255 if i % 8 < 4 else 0
            write(tail)
        for packet in stream.encode():
            output.mux(packet)
    result = analyze(Session(inspect_video(str(path)), rois))
    assert (result.test_start, result.test_end, result.range_confirmed) == (120, 1119, True)
    assert len(result.rounds) == 10
    assert [r.green.frame for r in result.rounds] == [170 + 100 * i for i in range(10)]
    assert all((r.md, r.tx) == (88.1, -0.7) for r in result.rounds)
    record = json.loads((Path(result.diagnostic_dir) / "marker.json").read_text())
    assert record["end"] == 1119
    with (Path(result.diagnostic_dir) / "signals.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1000 and rows[0]["frame"] == "120" and rows[-1]["frame"] == "1119"


@pytest.mark.parametrize("press_mode", ["reference", "sample"])
def test_calibrated_press_pipeline_works_without_hand_model_and_keeps_post_frames(
    tmp_path, monkeypatch, press_mode
):
    from zipfile import ZipFile

    from pcts_video_tool import analysis

    def no_hand_model():
        raise AssertionError("校准模式不应加载手部模型")

    monkeypatch.setattr(analysis, "create_hand_detector", no_hand_model)
    path = tmp_path / "calibrated.nut"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("ffv1", rate=240)
        stream.width, stream.height, stream.pix_fmt = 640, 360, "bgr0"
        for i in range(200):
            image = np.full((360, 640, 3), 230, np.uint8)
            image[9:81, 16:96] = 0 if 75 <= i < 79 or 90 <= i < 94 else 255
            cv2.putText(
                image,
                "PRESS HERE",
                (270, 83),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 170, 0) if 30 <= i < 45 or 100 <= i < 160 else (0, 0, 0),
                2,
            )
            if 103 <= i < 160:
                cv2.putText(
                    image,
                    "MD:88.1 TX:-0.7",
                    (215, 245),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 120, 0),
                    2,
                )
            gap = max(0, min(30, (60 - i) * 3)) if i <= 80 else (i - 80) * 3
            if i < 45:
                gap = max(0, min(30, (25 - i) * 3)) if i <= 35 else (i - 35) * 3
            half_width = 14 if 60 <= i <= 80 else 11
            cv2.rectangle(
                image,
                (100 + gap, 205 - half_width),
                (150 + gap, 205 + half_width),
                (100, 150, 200),
                -1,
            )
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts, frame.time_base = i, Fraction(1, 240)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    rois = {
        name: normalize_roi(roi, 640, 360)
        for name, roi in {
            "flash": [0.025, 0.025, 0.125, 0.2],
            "green": [0.4, 0.1, 0.55, 0.2],
            "reading": [0.32, 0.5, 0.65, 0.28],
            "touch": [0.1, 0.45, 0.25, 0.35],
        }.items()
    }
    session = Session(
        inspect_video(str(path)),
        rois,
        range_mode="manual",
        test_start=0,
        test_end=199,
        range_confirmed=True,
    )
    session.touch_calibration = TouchCalibration(
        65, [100 / 639, 170 / 359, 100 / 639, 250 / 359], [130 / 639, 205 / 359]
    )
    session.press_mode = press_mode
    if press_mode == "sample":
        session.touch_calibration.line = []
    result = analyze(session)
    assert len(result.rounds) == 1
    assert result.rounds[0].press.frame == 60 and result.rounds[0].press.state == "review"
    assert result.rounds[0].blink1.frame == 75
    root = Path(result.diagnostic_dir)
    decisions = json.loads((root / "touch-detection.json").read_text())
    if press_mode == "reference":
        assert decisions[0]["candidates"][0]["confirmed_at"] == 62
    else:
        assert decisions[0]["direction"][0] < -0.9
    with (root / "touch-signals.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert max(int(row["frame"]) for row in rows) == (87 if press_mode == "reference" else 99)
    assert (
        min(int(row["frame"]) for row in rows) == 45
    )  # Exclude press=25 from the orphan green response.
    bundle = tmp_path / "touch-diagnostics.zip"
    export_diagnostics(result, bundle)
    with ZipFile(bundle) as archive:
        if press_mode == "sample":
            assert {
                "analysis/touch-preview-01.png",
                "analysis/touch-targets.jsonl",
                "analysis/touch-signals.csv",
            }.issubset(archive.namelist())
            assert "analysis/touch-reference.png" not in archive.namelist()
            return
        assert {
            "analysis/touch-reference.png",
            "analysis/touch-reference-mask.png",
            "analysis/touch-calibration.json",
            "analysis/touch-detection.json",
            "analysis/touch-signals.csv",
        }.issubset(archive.namelist())
