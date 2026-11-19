from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean, stdev
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from .models import EVENTS, Session, calculate

MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
ET.register_namespace("", MAIN[1:-1])
ET.register_namespace("r", REL[1:-1])


def template_path() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "templates" / "AI_performance_test.xlsx"
    return Path(__file__).resolve().parents[2] / "docs" / "AI_performance_test.xlsx"


def write_template_report(session: Session, target: Path, status: str) -> None:
    """Fill this fixed workbook; preserve every unrelated ZIP member and the original file."""
    with ZipFile(template_path()) as original:
        book = ET.fromstring(original.read("xl/workbook.xml"))
        sheets = list(book.find(MAIN + "sheets"))
        selected = next((s for s in sheets if s.get("name") == "T1 (2)"), None)
        if selected is None:
            raise ValueError("Excel模板缺少T1 (2)工作表")
        relationships = ET.fromstring(original.read("xl/_rels/workbook.xml.rels"))
        location = next(
            r.get("Target") for r in relationships if r.get("Id") == selected.get(REL + "id")
        )
        location = location.lstrip("/") if location.startswith("/") else "xl/" + location
        sheet = ET.fromstring(original.read(location))
        rows = {int(r.get("r")): r for r in sheet.find(MAIN + "sheetData")}
        cells = {c.get("r"): c for row in rows.values() for c in row}
        strings = [
            "".join(t.text or "" for t in s.iter(MAIN + "t"))
            for s in ET.fromstring(original.read("xl/sharedStrings.xml"))
        ]
        for address, expected in {
            "A5": "press",
            "B5": "blink 1",
            "C5": "blink 2",
            "E5": "blink 1",
            "F5": "blink 2",
            "G5": "green",
            "I5": "MD",
            "J5": "TX",
        }.items():
            cell = cells.get(address)
            value = cell.find(MAIN + "v") if cell is not None else None
            if (
                cell is None
                or cell.get("t") != "s"
                or value is None
                or strings[int(value.text)].strip() != expected
            ):
                raise ValueError(f"Excel模板{address}应为{expected}，请恢复原模板布局")

        def write(address, value=None, formula=None):
            cell = cells.get(address)
            if cell is None:
                number = int("".join(c for c in address if c.isdigit()))
                cell = ET.SubElement(rows[number], MAIN + "c", r=address)
                cells[address] = cell
            cell.attrib.pop("t", None)
            for child in list(cell):
                if child.tag in (MAIN + "v", MAIN + "f", MAIN + "is"):
                    cell.remove(child)
            if formula is not None:
                ET.SubElement(cell, MAIN + "f").text = formula
                if value is None:
                    cell.set("t", "str")
                ET.SubElement(cell, MAIN + "v").text = None if value is None else str(value)
            elif isinstance(value, str):
                cell.set("t", "inlineStr")
                ET.SubElement(ET.SubElement(cell, MAIN + "is"), MAIN + "t").text = value
            elif value is not None:
                ET.SubElement(cell, MAIN + "v").text = str(value)

        note = "；其他工作表保留模板原内容" if len(sheets) > 1 else ""
        write("A2", f"{status}；当前{len(session.rounds)}/10轮；本次结果为T1 (2){note}。")
        write("B20", session.fps)
        write(
            "C20",
            "统计第1～10轮；"
            + (
                "按帧差/FPS计算；主表为原片帧号（从0开始）。"
                if session.method == "frames"
                else "按视频时间戳计算，B20不参与耗时计算；主表仍为原片帧号，请在工具中修改并重新导出。"
            ),
        )
        # PTS helper values stay out of the original visible layout. No frame number becomes seconds.
        columns = sheet.find(MAIN + "cols")
        ranges = []
        for column in columns:
            first, last = int(column.get("min")), int(column.get("max"))
            if last < 25 or first > 28:
                ranges.append(column)
                continue
            # The supplied template has a T:XFD span; split it instead of overlapping it.
            for start, end, hidden in (
                (first, min(last, 24), False),
                (max(first, 25), min(last, 28), True),
                (max(first, 29), last, False),
            ):
                if start > end:
                    continue
                attributes = dict(column.attrib, min=str(start), max=str(end))
                if hidden:
                    attributes.update(hidden="1", width="18", customWidth="1")
                ranges.append(ET.Element(MAIN + "col", attributes))
        columns[:] = ranges
        sheet.find(MAIN + "dimension").set("ref", "A2:AB121")
        summaries = {column: [] for column in ("I", "J", "O", "P", "Q", "R", "S", "T", "U", "V")}
        for index in range(10):
            row = index + 6
            item = session.rounds[index] if index < len(session.rounds) else None
            frames = [getattr(item, name).frame for name in EVENTS] if item else [None] * 4
            md, tx = (item.md, item.tx) if item else (None, None)
            times = [session.video.times[f] if f is not None else None for f in frames]
            for column, value in zip(
                ("A", "B", "C", "G", "I", "J", "Y", "Z", "AA", "AB"), (*frames, md, tx, *times)
            ):
                write(f"{column}{row}", value)
            for column, source, value in (("E", "B", frames[1]), ("F", "C", frames[2])):
                write(f"{column}{row}", value, f'IF(ISNUMBER({source}{row}),{source}{row},"")')
            aligned = frames[1] is not None and frames[2] is not None
            write(
                f"L{row}",
                0 if aligned else None,
                f'IF(COUNT(B{row},C{row},E{row},F{row})=4,(E{row}+F{row}-B{row}-C{row})/2,"")',
            )
            # One source video: missing blinks do not invalidate a known press or end-to-end time.
            write(f"M{row}", frames[0], f'IF(ISNUMBER(A{row}),A{row}+N(L{row}),"")')
            metric = calculate(session, item) if item else {}
            output = {
                column: metric.get(key)
                for column, key in (
                    ("O", "end_to_end"),
                    ("P", "hu_input"),
                    ("Q", "blink_interval"),
                    ("R", "hu_output"),
                    ("S", "md_total"),
                    ("T", "flash_accuracy"),
                    ("V", "hu_total"),
                )
            }
            output["U"] = 0 if aligned else None
            pairs = (
                (("M", "G"), ("M", "E"), ("E", "F"), ("F", "G"))
                if session.method == "frames"
                else (("Y", "AB"), ("Y", "Z"), ("Z", "AA"), ("AA", "AB"))
            )
            for column, (first, last) in zip(("O", "P", "Q", "R"), pairs):
                factor = "1000/$B$20" if session.method == "frames" else "1000"
                write(
                    f"{column}{row}",
                    output[column],
                    f'IF(COUNT({first}{row},{last}{row})=2,({last}{row}-{first}{row})*{factor},"")',
                )
            for column, first, operator, last in (
                ("S", "I", "+", "J"),
                ("T", "S", "-", "Q"),
                ("V", "O", "-", "S"),
            ):
                write(
                    f"{column}{row}",
                    output[column],
                    f'IF(COUNT({first}{row},{last}{row})=2,{first}{row}{operator}{last}{row},"")',
                )
            write(
                f"U{row}",
                output["U"],
                f'IF(COUNT(B{row},C{row},E{row},F{row})=4,((C{row}-B{row})-(F{row}-E{row}))*1000/$B$20,"")',
            )
            for column, value in {"I": md, "J": tx, **output}.items():
                if value is not None:
                    summaries[column].append(value)
        for column, data in summaries.items():
            extent = f"{column}6:{column}15"
            write(
                f"{column}17",
                mean(data) if data else None,
                f'IF(COUNT({extent})>0,AVERAGE({extent}),"")',
            )
            write(
                f"{column}18",
                stdev(data) if len(data) > 1 else None,
                f'IF(COUNT({extent})>1,STDEV({extent}),"")',
            )
        book.find(MAIN + "bookViews/" + MAIN + "workbookView").set(
            "activeTab", str(sheets.index(selected))
        )
        book.find(MAIN + "calcPr").attrib.update(
            calcMode="auto", fullCalcOnLoad="1", forceFullCalc="1"
        )
        replacements = {
            location: ET.tostring(sheet, encoding="utf-8", xml_declaration=True),
            "xl/workbook.xml": ET.tostring(book, encoding="utf-8", xml_declaration=True),
        }
        with ZipFile(target, "w") as output:
            for member in original.infolist():
                output.writestr(
                    member, replacements.get(member.filename, original.read(member.filename))
                )
