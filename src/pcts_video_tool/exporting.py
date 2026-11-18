from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev

import cv2
import xlsxwriter

from .models import EVENTS, METRICS, Session, calculate, check_source
from .video import rate_summary, read_frame


def export_report(
    session: Session,
    destination: Path,
    evidence: bool = True,
    cancelled=lambda: False,
    progress=lambda message: None,
) -> Path:
    check_source(session.video)
    destination.mkdir(parents=True, exist_ok=False)
    problems = session.problems()
    status = "草稿：存在待确认或缺失数据" if problems else "数据已确认（不包含合格判定）"
    target = destination / ("测量结果-草稿.xlsx" if problems else "测量结果.xlsx")
    temporary = destination / "writing.xlsx"
    with xlsxwriter.Workbook(temporary) as book:
        header = book.add_format({"bold": True, "bg_color": "#17324D", "font_color": "white"})
        number = book.add_format({"num_format": "0.00"})
        title = book.add_format({"bold": True, "font_size": 14})
        warn = book.add_format({"font_color": "#9A5300", "text_wrap": True})
        sheet = book.add_worksheet("测量结果")
        sheet.freeze_panes(5, 1)
        sheet.set_column(0, 4, 12)
        sheet.set_column(5, 13, 17)
        sheet.set_column(14, 14, 70)
        sheet.write(0, 0, "PCTS · 视频测量结果", title)
        sheet.write(1, 0, status, warn if problems else title)
        method = "帧差 / FPS" if session.method == "frames" else "视频展示时间戳差"
        sheet.write(2, 0, f"计算方式：{method}；FPS设置：{session.fps:g}；帧号从0开始")
        sheet.write(
            3,
            0,
            f"预期10轮 / 当前{len(session.rounds)}轮；有效帧{session.test_start}～{session.test_end}；MD/TX单位ms；全部轮次参与统计",
        )
        columns = [
            "轮次",
            "press",
            "blink 1",
            "blink 2",
            "green",
            "MD",
            "TX",
            *[name + " (ms)" for name in METRICS.values()],
            "复核状态",
        ]
        sheet.write_row(4, 0, columns, header)
        timeline = book.add_worksheet("事件时间与来源")
        timeline.set_column(0, 0, 9)
        timeline.set_column(1, 4, 18)
        timeline.set_column(5, 10, 32)
        timeline.write_row(
            0,
            0,
            [
                "轮次",
                *[n + " 秒" for n in EVENTS],
                *[n + " 来源" for n in EVENTS],
                "MD/TX来源帧",
                "OCR原文",
            ],
            header,
        )
        values = []
        for i, item in enumerate(session.rounds):
            if cancelled():
                raise InterruptedError("已取消导出，未生成最终报告")
            row, excel_row, tr = i + 5, i + 6, i + 2
            events = [getattr(item, name) for name in EVENTS]
            sheet.write_row(row, 0, [i + 1, *[e.frame for e in events], item.md, item.tx])
            times = [None if e.frame is None else session.video.times[e.frame] for e in events]
            timeline.write_row(
                i + 1,
                0,
                [i + 1, *times, *[e.state for e in events], item.reading_frame, item.ocr_text],
            )
            metric = calculate(session, item)
            values.append(metric)
            if session.method == "frames":
                factor = f"1000/{session.fps!r}"
                formulas = [
                    f"(E{excel_row}-B{excel_row})*{factor}",
                    f"(C{excel_row}-B{excel_row})*{factor}",
                    f"(D{excel_row}-C{excel_row})*{factor}",
                    f"(E{excel_row}-D{excel_row})*{factor}",
                ]
            else:
                prefix = "'事件时间与来源'!"
                formulas = [
                    f"({prefix}{b}{tr}-{prefix}{a}{tr})*1000"
                    for a, b in (("B", "E"), ("B", "C"), ("C", "D"), ("D", "E"))
                ]
            formulas.extend(
                [
                    f"F{excel_row}+G{excel_row}",
                    f"H{excel_row}-L{excel_row}",
                    f"L{excel_row}-J{excel_row}",
                ]
            )
            for col, (key, formula) in enumerate(zip(METRICS, formulas), 7):
                if metric[key] is not None:
                    sheet.write_formula(row, col, "=" + formula, number, metric[key])
            sheet.write(
                row, 14, "；".join(item.problems()) or "已确认", warn if item.problems() else None
            )
        summary = len(session.rounds) + 7
        for j, label in enumerate(("均值", "样本标准差", "有数值轮数")):
            sheet.write(summary + j, 0, label, header)
        for col, key in enumerate(METRICS, 7):
            data = [v[key] for v in values if v[key] is not None]
            if data:
                sheet.write_number(summary, col, mean(data), number)
            if len(data) > 1:
                sheet.write_number(summary + 1, col, stdev(data), number)
            sheet.write_number(summary + 2, col, len(data))
        notes = book.add_worksheet("说明与视频检查")
        notes.set_column(0, 0, 115)
        texts = [
            status,
            f"原视频：{session.video.path}",
            rate_summary(session.video, session.fps),
            "帧数模式假设等间隔；时间戳模式使用文件展示时间轴，变速素材需核对采集时间。",
            "统计包含所有有数值的轮次，不排除第一轮。草稿统计仅供预览，有数值不等于已确认。",
            "Excel公式含生成时缓存。若直接在Excel修改输入，统计需重新导出；建议在工具内修正。",
            *session.analysis_notes,
            *problems,
        ]
        for i, text in enumerate(texts):
            notes.write(i, 0, text)
    if evidence:
        for i, item in enumerate(session.rounds, 1):
            progress(f"导出第{i}/{len(session.rounds)}轮证据")
            folder = destination / "evidence" / f"round-{i:02d}"
            folder.mkdir(parents=True)
            frames = {name: getattr(item, name).frame for name in EVENTS}
            frames["md-tx"] = item.reading_frame
            for name, index in frames.items():
                if cancelled():
                    raise InterruptedError("已取消导出，未生成最终报告")
                if index is not None:
                    image = read_frame(session.video, index)
                    ok, encoded = cv2.imencode(".png", image)
                    if not ok:
                        raise OSError("证据图像编码失败")
                    (folder / f"{name}-frame-{index}.png").write_bytes(encoded.tobytes())
    (destination / "results.json").write_text(
        json.dumps(
            {"status": status, "session": asdict(session), "metrics": values},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(target)
    return target
