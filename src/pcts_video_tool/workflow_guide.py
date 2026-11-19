"""Contextual instructions for importing, selecting regions, and reviewing a run."""

from html import escape
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QTabWidget, QTextBrowser, QVBoxLayout

from .models import Session
from .run_logs import log_directory

REGIONS = {"flash": "灯光区域", "green": "变绿文字", "reading": "MD / TX", "touch": "触摸观察区域"}
EXAMPLES = Path(__file__).parent / "assets" / "guide"
REGION_TIPS = {
    "flash": "矩形贴住发光点，框内避开旁边的反光。",
    "green": "沿文字边缘框全 PRESS HERE，留少量空隙，避开屏幕黑边。",
    "reading": "同框包含 MD、TX 和完整数值，保留小数点及负号。",
    "touch": "箭头为接近方向，橙圈为第 1941 帧的落点附近。紫框留出指腹接近路径和落点前方的屏幕。",
}


def guide_stage(session: Session | None) -> str:
    if session is None:
        return "import"
    if session.rounds:
        return "review"
    if REGIONS.keys() - session.rois.keys():
        return "regions"
    return "empty" if session.analysis_notes else "ready"


def _picture(key, width, height, alt):
    url = QUrl.fromLocalFile(str(EXAMPLES / f"{key}.png")).toString()
    return f'<img src="{escape(url, quote=True)}" width="{width}" height="{height}" alt="{alt}">'


def _cards(entries, columns):
    """Lay out instruction cards using the table subset supported by Qt rich text."""
    rows = []
    for start in range(0, len(entries), columns):
        cells = "".join(
            f'<td width="{100 // columns}%" bgcolor="#f4f7f5" valign="top">{entry}</td>'
            for entry in entries[start : start + columns]
        )
        rows.append(f"<tr>{cells}</tr>")
    return '<table width="99%" cellspacing="6" cellpadding="9">' + "".join(rows) + "</table>"


def _page(stage, title, summary, body):
    browser = QTextBrowser()
    browser.setStyleSheet("QTextBrowser { background:white; border:0; }")
    browser.document().setDocumentMargin(12)
    steps = (
        ("import", "拍摄与导入"),
        ("regions", "框选区域"),
        ("ready", "自动分析"),
        ("review", "人工复核"),
    )
    steps_html = ""
    if stage:
        labels = []
        for key, label in steps:
            active = key == stage or (key == "ready" and stage == "empty")
            bg, fg = ("#157347", "#ffffff") if active else ("#e9f1ed", "#445e53")
            labels.append(
                f'<td width="25%" align="center" bgcolor="{bg}"><font color="{fg}"><b>{label}</b></font></td>'
            )
        steps_html = (
            '<table width="99%" cellspacing="3" cellpadding="5"><tr>'
            + "".join(labels)
            + "</tr></table>"
        )
    browser.setHtml(
        "<style>body, p, td {font-size:14px; color:#243e33} h2 {font-size:22px; color:#143e2c; margin:0 0 5px}"
        "p {margin:5px 0} th {font-size:15px} .card-title {font-size:16px; color:#163c2e}</style>"
        + steps_html
        + f'<table width="99%" cellspacing="6" cellpadding="2"><tr><td><h2 style="font-size:22px">{title}</h2>'
        f"<p>{summary}</p></td></tr></table>{body}"
    )
    return browser


def _examples(session, selecting):
    cards = []
    for key, name in REGIONS.items():
        status = ""
        if selecting:
            label, color = ("已框选", "#276955") if key in session.rois else ("待框选", "#86521a")
            status = f'&nbsp;&nbsp;<small style="color:{color}">{label}</small>'
        cards.append(
            f'<b class="card-title">{name}</b>{status}<br>'
            + _picture(key, 280, 135, f"{name}框选示例")
            + f"<p>{REGION_TIPS[key]}</p>"
        )
    title = f"框选四个区域（已完成 {len(session.rois)}/4）" if selecting else "四种区域的框选示例"
    return _page(
        "regions" if selecting else None,
        title,
        "灯光用矩形拖选。其余按左上、右上、右下、左下依次点四角，框好后可拖动角点微调。",
        _cards(cards, 2),
    )


class DetectionGuide(QDialog):
    def __init__(self, session, colors, parent=None):
        super().__init__(parent)
        self.setWindowTitle("操作指引")
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane { background:white; border:1px solid #cfddd5; }
            QTabBar::tab { padding:7px 14px; font-size:13px; color:#4a6256; }
            QTabBar::tab:selected { color:#12683f; background:white; border-bottom:2px solid #188552; }
        """)
        layout.addWidget(tabs)
        stage = guide_stage(session)
        action_text = None
        if stage == "import":
            cards = [
                _picture("steady-camera", 245, 77, "使用支架固定拍摄设备")
                + '<p><b class="card-title">固定拍摄机位</b></p><p>用支架固定相机或拍摄手机，从第一轮到最后一轮保持相同的位置和角度。</p>',
                _picture("fixed-tap", 245, 77, "每轮点击屏幕的同一落点")
                + '<p><b class="card-title">固定点击位置</b></p><p>固定被测手机，选准屏幕上的一个落点，每一轮都点击同一位置。</p>',
            ]
            current = _page(
                stage,
                "导入原始视频",
                "按下图准备拍摄。录制结束后，点击下方“导入视频”选择原片。",
                _cards(cards, 2) + "<p>继续上次复核时，可从主窗口选择“打开任务”。</p>",
            )
            action_text = "导入视频…"
        elif stage == "regions":
            current = _examples(session, selecting=True)
            first = next(name for key, name in REGIONS.items() if key not in session.rois)
            action_text = f"去框选：{first}"
        elif stage == "ready":
            checks = [
                f'<font color="#157347"><b>✓</b></font>&nbsp; {name}' for name in REGIONS.values()
            ]
            current = _page(
                stage,
                "开始自动分析",
                "四个选区已就绪，点击下方按钮开始。",
                _cards(checks, 2)
                + "<p>主窗口会显示分析进度和耗时。分析结束后，点击轮次表核对关键帧和读数。</p>",
            )
            action_text = "开始自动分析"
        elif stage == "empty":
            current = _page(
                stage,
                "检查选区与有效片段",
                "当前没有可复核的轮次。",
                _cards(
                    [
                        '<b class="card-title">查看检测记录</b><p>切到“检测记录”页，查看本次分析给出的原因。</p>',
                        '<b class="card-title">回看原片</b><p>核对灯光、变绿文字和“有效片段”的起止帧，修正后重新分析。</p>',
                    ],
                    2,
                ),
            )
            action_text = "检查变绿文字选区"
        else:
            legend = "".join(
                f'<td width="33%" align="center" bgcolor="{colors[key]}"><b>{word}</b>&nbsp; {meaning}</td>'
                for key, word, meaning in (
                    ("missing", "红", "缺失"),
                    ("review", "黄", "待复核"),
                    ("accepted", "绿", "自动通过或人工确认"),
                )
            )
            cards = [
                '<b class="card-title">事件帧</b><p>单击 press、blink 或 green 查看画面，用左右键逐帧定位。'
                "右键该格可查看候选帧，或选“确认当前帧”写入视频区显示的帧号。</p>",
                '<b class="card-title">MD / TX 读数</b><p>双击数值格编辑，回车保存。核对后右键确认该项读数。</p>',
            ]
            if any(r.green_recommendation for r in session.rounds):
                cards.append(
                    '<b class="card-title">green 推荐帧</b><p>选中带“荐”标记的轮次查看推荐帧。'
                    "点击“采用推荐帧”才会更新测量值。</p>"
                )
            current = _page(
                stage,
                "逐轮复核并微调",
                f"当前有 {len(session.rounds)} 轮，优先处理红色和黄色格。",
                f'<table width="99%" cellspacing="6" cellpadding="7"><tr>{legend}</tr></table>'
                + _cards(cards, 1)
                + "<p>复核后点击“保存任务”，或“导出 Excel + 证据”。</p>",
            )
        tabs.addTab(current, "当前步骤")
        if session is not None and stage != "regions":
            tabs.addTab(_examples(session, selecting=False), "框选示例")
        notes = QTextBrowser()
        entries = session.analysis_notes + session.video.warnings if session is not None else []
        notes.setPlainText(
            "\n\n".join([*(entries or ["暂无检测记录。"]), f"诊断日志位置：{log_directory()}"])
        )
        tabs.addTab(notes, "检测记录")

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.button(QDialogButtonBox.Close).setText("关闭")
        buttons.rejected.connect(self.reject)
        if action_text:
            primary = buttons.addButton(action_text, QDialogButtonBox.AcceptRole)
            primary.setDefault(True)
            primary.setStyleSheet(
                "background:#157347; color:white; border-radius:4px; padding:7px 14px; font-weight:700"
            )
            buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
        available = self.screen().availableGeometry()
        height = {"import": 450, "regions": 690, "review": 505, "empty": 355, "ready": 355}[stage]
        self.resize(min(730, available.width() - 32), min(height, available.height() - 60))
        if session is not None and stage != "regions":

            def fit_examples(index):
                if index == 1:
                    self.resize(self.width(), max(self.height(), min(690, available.height() - 60)))

            tabs.currentChanged.connect(fit_examples)
