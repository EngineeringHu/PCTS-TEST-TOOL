from __future__ import annotations

import copy
import json
import math
from datetime import datetime
from pathlib import Path
from time import perf_counter

from PySide6.QtCore import QPointF, QRect, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QKeySequence,
    QPainter,
    QPalette,
    QPen,
    QPixmap,
    QPolygonF,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from . import __version__
from .analysis import analyze
from .exporting import export_report
from .models import (
    EVENTS,
    Event,
    Round,
    Session,
    TouchCalibration,
    calculate,
    check_source,
    load_session,
    normalize_roi,
    save_session,
    validate_touch_calibration,
)
from .motion import MotionTracker, motion_preview
from .region_frames import prepare_region_session
from .run_logs import export_diagnostics, log_directory, logger
from .touch import ContactTracker
from .video import FrameReader, crop, rate_summary, read_frame

REGIONS = {"flash": "灯光区域", "green": "变绿文字", "reading": "MD / TX", "touch": "触摸观察区域"}
PRESS_MODES = {
    "auto": "指腹设置：自动",
    "sample": "指腹设置：单点",
    "reference": "指腹设置：参考线",
}
COLORS = {"flash": "#f59e0b", "green": "#22c55e", "reading": "#38bdf8", "touch": "#c084fc"}
REVIEW_REGIONS = {
    "press": "touch",
    "blink1": "flash",
    "blink2": "flash",
    "green": "green",
    "reading": "reading",
}
CELL_COLORS = {"accepted": "#dcfce7", "review": "#fef3c7", "missing": "#fee2e2"}


class ResultTable(QTableWidget):
    def mousePressEvent(self, event):
        # Opening a menu must preserve the frame the user is about to confirm.
        if event.button() != Qt.RightButton:
            super().mousePressEvent(event)


class ResultCellDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        option = QStyleOptionViewItem(option)
        background = index.data(Qt.BackgroundRole)
        if background is not None:
            option.palette.setColor(QPalette.Highlight, background.color().darker(110))
            option.palette.setColor(QPalette.HighlightedText, QColor("#1e293b"))
        super().paint(painter, option, index)


class Worker(QThread):
    progress = Signal(str)
    percentage = Signal(int)
    duration = Signal(float)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.function = function

    def run(self):
        started = perf_counter()
        try:
            result = self.function(
                self.progress.emit, self.isInterruptionRequested, self.percentage.emit
            )
        except InterruptedError:
            self.duration.emit(perf_counter() - started)
            self.failed.emit("操作已取消，原有任务保留")
        except Exception as error:
            self.duration.emit(perf_counter() - started)
            # Desktop entry boundary: present unexpected library/IO errors without losing work.
            logger.exception("后台任务失败")
            self.failed.emit(f"{type(error).__name__}: {error}")
        else:
            self.duration.emit(perf_counter() - started)
            self.percentage.emit(100)
            self.completed.emit(result)


class VideoCanvas(QWidget):
    region_changed = Signal(str, list)
    selection_error = Signal(str)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(480, 280)
        self.setFocusPolicy(Qt.StrongFocus)
        self.pixmap = QPixmap()
        self.empty_text = "导入原始视频，开始逐帧测量"
        self.rois = {}
        self.regions_visible = True
        self.selection = None
        self.shape = "quad"
        self.origin = None
        self.cursor_point = None
        self.pending = []
        self.dragging = None
        self.preview = None

    def set_image(self, image):
        rgb = image[:, :, ::-1].copy()
        qimage = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qimage.copy())
        self.update()

    def clear_image(self, message):
        self.pixmap = QPixmap()
        self.empty_text = message
        self.update()

    def image_rect(self):
        if self.pixmap.isNull():
            return QRect()
        size = self.pixmap.size().scaled(self.size(), Qt.KeepAspectRatio)
        return QRect(
            (self.width() - size.width()) // 2,
            (self.height() - size.height()) // 2,
            size.width(),
            size.height(),
        )

    def reset_selection(self):
        self.origin = self.cursor_point = self.dragging = self.preview = None
        self.pending = []
        self.update()

    def set_region(self, name):
        self.reset_selection()
        self.selection = name

    def set_shape(self, shape):
        self.reset_selection()
        self.shape = shape

    def set_regions_visible(self, visible):
        self.regions_visible = visible
        self.reset_selection()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#111d2b"))
        rect = self.image_rect()
        if self.pixmap.isNull():
            painter.setPen(QColor("#bbc8d6"))
            painter.drawText(self.rect(), Qt.AlignCenter, self.empty_text)
            return
        painter.drawPixmap(rect, self.pixmap)
        if not self.regions_visible:
            return
        for name, stored in self.rois.items():
            roi = self.preview if name == self.selection and self.preview is not None else stored
            points = [
                QPointF(rect.x() + x * rect.width(), rect.y() + y * rect.height())
                for x, y in zip(roi[::2], roi[1::2])
            ]
            painter.setPen(QPen(QColor(COLORS[name]), 2))
            painter.drawPolygon(QPolygonF(points))
            painter.drawText(points[0] + QPointF(3, -5), REGIONS[name])
            if name == self.selection:
                for i, point in enumerate(points, 1):
                    painter.drawEllipse(point, 5, 5)
                    painter.drawText(point + QPointF(6, 13), str(i))
        if self.pending:
            points = [
                QPointF(rect.x() + x * rect.width(), rect.y() + y * rect.height())
                for x, y in self.pending
            ]
            painter.setPen(QPen(QColor("white"), 2))
            painter.drawPolyline(QPolygonF(points))
            for i, point in enumerate(points, 1):
                painter.drawEllipse(point, 4, 4)
                painter.drawText(point + QPointF(5, -6), str(i))
        if self.origin is not None and self.cursor_point is not None:
            painter.setPen(QPen(QColor("white"), 2, Qt.DashLine))
            painter.drawRect(QRect(self.origin, self.cursor_point).normalized())

    def emit_roi(self, values):
        try:
            roi = normalize_roi(values, self.pixmap.width(), self.pixmap.height())
        except ValueError as error:
            self.selection_error.emit(str(error))
        else:
            self.region_changed.emit(self.selection, roi)
        self.reset_selection()

    def mousePressEvent(self, event):
        if not self.regions_visible:
            return
        if event.button() == Qt.RightButton:
            if self.pending:
                self.pending.pop()
                self.update()
            else:
                self.reset_selection()
            return
        rect = self.image_rect()
        if not self.selection or not rect.contains(event.position().toPoint()):
            return
        self.setFocus()
        point = event.position()
        if self.shape == "rectangle":
            self.origin = point.toPoint()
            self.cursor_point = self.origin
            return
        roi = self.rois.get(self.selection)
        if roi and not self.pending:
            for i, (x, y) in enumerate(zip(roi[::2], roi[1::2])):
                if (point.x() - rect.x() - x * rect.width()) ** 2 + (
                    point.y() - rect.y() - y * rect.height()
                ) ** 2 <= 100:
                    self.dragging, self.preview = i, list(roi)
                    return
        self.pending.append(
            ((point.x() - rect.x()) / rect.width(), (point.y() - rect.y()) / rect.height())
        )
        if len(self.pending) == 4:
            self.emit_roi([v for point in self.pending for v in point])
        self.update()

    def mouseMoveEvent(self, event):
        if self.dragging is not None:
            rect = self.image_rect()
            x = max(0.0, min(1.0, (event.position().x() - rect.x()) / rect.width()))
            y = max(0.0, min(1.0, (event.position().y() - rect.y()) / rect.height()))
            self.preview[self.dragging * 2 : self.dragging * 2 + 2] = [x, y]
        elif self.origin is not None:
            self.cursor_point = event.position().toPoint()
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        if self.dragging is not None:
            self.mouseMoveEvent(event)
            self.emit_roi(self.preview)
        elif self.origin is not None:
            rect = self.image_rect()
            box = QRect(self.origin, event.position().toPoint()).normalized().intersected(rect)
            if box.width() >= 6 and box.height() >= 6:
                self.emit_roi(
                    [
                        (box.x() - rect.x()) / rect.width(),
                        (box.y() - rect.y()) / rect.height(),
                        box.width() / rect.width(),
                        box.height() / rect.height(),
                    ]
                )
            else:
                self.reset_selection()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.reset_selection()
        else:
            super().keyPressEvent(event)


class FrameComparison(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        self.title = QLabel("三帧对照")
        self.title.setWordWrap(True)
        bar.addWidget(self.title, 1)
        self.crop_only = QCheckBox("仅看选区")
        self.crop_only.setChecked(True)
        bar.addWidget(self.crop_only)
        self.back_button = QPushButton("返回全画面")
        bar.addWidget(self.back_button)
        layout.addLayout(bar)
        columns = QHBoxLayout()
        self.boxes, self.panes = [], []
        self.frame_indices = [None, None, None]
        for i, label in enumerate(("上一帧", "当前帧", "下一帧")):
            box = QGroupBox(label)
            if i == 1:
                box.setStyleSheet(
                    "QGroupBox { border:2px solid #176b87; } QGroupBox::title { color:#176b87; font-weight:bold; }"
                )
            pane = VideoCanvas()
            pane.setMinimumSize(160, 160)
            pane.set_regions_visible(False)
            QVBoxLayout(box).addWidget(pane)
            columns.addWidget(box, 1)
            self.boxes.append(box)
            self.panes.append(pane)
        layout.addLayout(columns, 1)

    def set_panel(self, position, frame, image, message=""):
        self.frame_indices[position] = frame
        label = ("上一帧", "当前帧", "下一帧")[position]
        self.boxes[position].setTitle(f"{label} · {frame if frame is not None else '无'}")
        if image is None:
            self.panes[position].clear_image(message)
        else:
            self.panes[position].set_image(image)


class CalibrationCanvas(VideoCanvas):
    points_changed = Signal()

    def __init__(self):
        super().__init__()
        self.points = []
        self.max_points = 3

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            if self.points:
                self.points.pop()
        elif event.button() == Qt.LeftButton and self.max_points:
            if self.max_points == 1:
                self.points = []
            if len(self.points) >= self.max_points:
                return
            rect = self.image_rect()
            point = event.position()
            if not rect.contains(point.toPoint()):
                return
            self.points.append(
                [
                    max(0.0, min(1.0, (point.x() - rect.x()) / (rect.width() - 1))),
                    max(0.0, min(1.0, (point.y() - rect.y()) / (rect.height() - 1))),
                ]
            )
        self.update()
        self.points_changed.emit()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        rect = self.image_rect()
        points = [
            QPointF(rect.x() + x * (rect.width() - 1), rect.y() + y * (rect.height() - 1))
            for x, y in self.points
        ]
        painter.setPen(QPen(QColor("#ff5c5c"), 2))
        if len(points) >= 2:
            painter.drawLine(points[0], points[1])
        for i, point in enumerate(points):
            painter.setPen(QPen(QColor("#4de5ff" if i == 2 else "#ff5c5c"), 2))
            painter.drawEllipse(point, 4, 4)
            painter.drawText(
                point + QPointF(6, -6), "指腹取样" if i == 2 or self.max_points == 1 else str(i + 1)
            )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"PCTS Video Tool {__version__} · 触屏耗时测量")
        self.resize(1440, 960)
        self.session = None
        self.frame_reader = None
        self.region_previews = {}
        self.worker = None
        self.current_frame = -1
        self.dirty = False
        self.editing = False
        self.comparison_field = None
        self.analysis_started = None
        self.analysis_elapsed_seconds = 0.0
        self.root = QWidget()
        self.setCentralWidget(self.root)
        layout = QVBoxLayout(self.root)
        bar = QHBoxLayout()
        title = QLabel("PCTS  /  VIDEO MEASUREMENT")
        title.setStyleSheet("font-size:20px; font-weight:700; color:#17324d; padding:8px")
        bar.addWidget(title)
        bar.addStretch()
        self.buttons = []
        for label, callback in (
            ("导入视频", self.open_video),
            ("打开任务", self.open_task),
            ("保存任务", self.save_task),
            ("导出 Excel + 证据", self.export),
            ("导出诊断包", self.export_debug),
            ("检测说明", self.show_analysis_notes),
            ("有效片段", self.configure_range),
            ("指腹设置：自动", self.configure_touch),
        ):
            button = QPushButton(label)
            button.clicked.connect(callback)
            bar.addWidget(button)
            self.buttons.append(button)
            if label == "指腹设置：自动":
                self.calibration_button = button
        layout.addLayout(bar)
        self.metadata = QLabel("本地分析 · 原片逐帧编号 · 预期10轮 · 两种计时方式")
        self.metadata.setWordWrap(True)
        layout.addWidget(self.metadata)
        split = QSplitter(Qt.Horizontal)
        left, right = QWidget(), QWidget()
        lv, rv = QVBoxLayout(left), QVBoxLayout(right)
        lv.setContentsMargins(0, 0, 8, 0)
        rv.setContentsMargins(4, 0, 0, 0)
        self.canvas = VideoCanvas()
        self.round_hint = QLabel()
        self.round_hint.setWordWrap(True)
        self.round_hint.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.round_hint.setStyleSheet("background: #213342; color: white; padding: 8px;")
        overlay = QVBoxLayout(self.canvas)
        overlay.addWidget(self.round_hint, 0, Qt.AlignTop | Qt.AlignHCenter)
        overlay.addStretch()
        self.round_hint.hide()
        self.round_hint_timer = QTimer(self)
        self.round_hint_timer.setSingleShot(True)
        self.round_hint_timer.timeout.connect(self.round_hint.hide)
        self.canvas.region_changed.connect(self.region_changed)
        self.canvas.selection_error.connect(self.statusBar().showMessage)
        self.comparison = FrameComparison()
        self.video_views = QStackedWidget()
        self.video_views.addWidget(self.canvas)
        self.video_views.addWidget(self.comparison)
        self.comparison.back_button.clicked.connect(lambda: self.review_layout.setCurrentIndex(1))
        self.comparison.crop_only.toggled.connect(lambda: self.show_frame(self.current_frame))
        lv.addWidget(self.video_views, 1)
        self.review_hint = QLabel()
        self.review_hint.setWordWrap(True)
        self.review_hint.hide()
        lv.addWidget(self.review_hint)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.slider_changed)
        lv.addWidget(self.slider)
        navigation = QHBoxLayout()
        for label, delta in (("−10帧", -10), ("上一帧", -1), ("下一帧", 1), ("+10帧", 10)):
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, d=delta: self.step(d))
            navigation.addWidget(button)
        navigation.addWidget(QLabel("原片帧号"))
        self.frame_input = QSpinBox()
        self.frame_input.setRange(-1, 0)
        self.frame_input.setSpecialValueText("未指定")
        self.frame_input.setValue(-1)
        self.frame_input.valueChanged.connect(self.show_frame)
        navigation.addWidget(self.frame_input)
        self.time_label = QLabel("—")
        navigation.addWidget(self.time_label)
        self.review_layout = QComboBox()
        self.review_layout.addItem("三帧对照", "comparison")
        self.review_layout.addItem("单幅全画面", "full")
        self.review_layout.currentIndexChanged.connect(self.change_review_layout)
        navigation.addWidget(self.review_layout)
        lv.addLayout(navigation)
        roi_bar = QHBoxLayout()
        self.region_combo = QComboBox()
        self.region_combo.addItem("查看画面（不框选）", None)
        for key, text in REGIONS.items():
            self.region_combo.addItem("框选：" + text, key)
        self.region_combo.currentIndexChanged.connect(self.select_region)
        self.region_combo.activated.connect(self.select_region)
        roi_bar.addWidget(self.region_combo)
        self.shape_combo = QComboBox()
        self.shape_combo.addItem("四点框选（侧拍）", "quad")
        self.shape_combo.addItem("矩形拖选", "rectangle")
        self.shape_combo.currentIndexChanged.connect(
            lambda: self.canvas.set_shape(self.shape_combo.currentData())
        )
        roi_bar.addWidget(self.shape_combo)
        self.region_visibility = QCheckBox("显示选区")
        self.region_visibility.setChecked(True)
        self.region_visibility.toggled.connect(self.canvas.set_regions_visible)
        roi_bar.addWidget(self.region_visibility)
        self.roi_status = QLabel("区域 0 / 4")
        roi_bar.addWidget(self.roi_status)
        self.run_button = QPushButton("开始自动分析")
        self.run_button.setStyleSheet("background:#176b87; color:white; font-weight:600")
        self.run_button.clicked.connect(self.start_analysis)
        roi_bar.addWidget(self.run_button)
        lv.addLayout(roi_bar)
        hint = QLabel(
            "四点顺序：左上→右上→右下→左下。可拖角点微调；右键撤销，Esc取消。左右键逐帧。"
        )
        hint.setWordWrap(True)
        lv.addWidget(hint)
        timing = QGroupBox("耗时计算")
        tf = QFormLayout(timing)
        self.method = QComboBox()
        self.method.addItem("按帧数 / FPS", "frames")
        self.method.addItem("按视频时间戳", "timestamps")
        self.fps = QDoubleSpinBox()
        self.fps.setRange(1, 10000)
        self.fps.setDecimals(3)
        self.fps.setValue(240)
        self.method.currentIndexChanged.connect(self.timing_changed)
        self.fps.valueChanged.connect(self.timing_changed)
        tf.addRow("计算方式", self.method)
        tf.addRow("FPS", self.fps)
        notice = QLabel(
            "切换后重算全部轮次。时间戳使用文件展示时间轴；变速素材需核对真实采集时间。"
        )
        notice.setWordWrap(True)
        tf.addRow(notice)
        rv.addWidget(timing)
        self.round_title = QLabel("选择下方轮次，复核或修正")
        self.round_title.setStyleSheet("font-weight:700; font-size:16px; padding:6px")
        rv.addWidget(self.round_title)
        fields = QFormLayout()
        self.event_inputs = {}
        for name in EVENTS:
            row = QHBoxLayout()
            spin = QSpinBox()
            spin.setRange(-1, 0)
            spin.setValue(-1)
            spin.setSpecialValueText("未识别")
            self.event_inputs[name] = spin
            spin.valueChanged.connect(lambda value, n=name: self.edit_event(n, value))
            row.addWidget(spin)
            jump = QPushButton("查看")
            jump.clicked.connect(lambda checked=False, n=name: self.jump_event(n))
            row.addWidget(jump)
            choose = QPushButton("设为当前帧")
            choose.clicked.connect(lambda checked=False, n=name: self.select_current(n))
            row.addWidget(choose)
            fields.addRow(name, row)
        self.green_options = QWidget()
        options = QHBoxLayout(self.green_options)
        options.setContentsMargins(0, 0, 0, 0)
        self.preview_green = QPushButton()
        self.preview_green.clicked.connect(self.preview_recommended_green)
        self.apply_green = QPushButton()
        self.apply_green.clicked.connect(self.apply_or_restore_green)
        options.addWidget(self.preview_green)
        options.addWidget(self.apply_green)
        fields.addRow(self.green_options)
        self.green_options.hide()
        self.md, self.tx = QLineEdit(), QLineEdit()
        self.md.setPlaceholderText("毫秒，可留空")
        self.tx.setPlaceholderText("毫秒，可为负数")
        self.md.textEdited.connect(lambda text: self.edit_number("md", text))
        self.tx.textEdited.connect(lambda text: self.edit_number("tx", text))
        fields.addRow("MD", self.md)
        fields.addRow("TX", self.tx)
        self.reading_frame = QSpinBox()
        self.reading_frame.setRange(-1, 0)
        self.reading_frame.setSpecialValueText("未指定")
        self.reading_frame.valueChanged.connect(self.edit_reading_frame)
        reading_row = QHBoxLayout()
        reading_row.addWidget(self.reading_frame)
        show_reading = QPushButton("查看读数帧")
        show_reading.clicked.connect(lambda: self.review_event("reading"))
        reading_row.addWidget(show_reading)
        use_reading = QPushButton("设为当前帧")
        use_reading.clicked.connect(lambda: self.select_current("reading"))
        reading_row.addWidget(use_reading)
        fields.addRow("读数来源", reading_row)
        rv.addLayout(fields)
        touch_evidence = QPushButton("查看本轮触摸依据")
        touch_evidence.clicked.connect(self.show_touch_evidence)
        rv.addWidget(touch_evidence)
        confirm = QPushButton("应用修改并确认本轮")
        confirm.clicked.connect(self.confirm_round)
        confirm.setStyleSheet("background:#176b87; color:white; padding:9px")
        rv.addWidget(confirm)
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        rv.addWidget(self.details, 1)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([950, 490])
        layout.addWidget(split, 3)
        self.table = ResultTable(0, 9)
        self.table.setHorizontalHeaderLabels(
            ["轮次", "press", "blink 1", "blink 2", "green", "MD", "TX", "端到端 ms", "复核状态"]
        )
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.EditKeyPressed)
        self.table.setItemDelegate(ResultCellDelegate(self.table))
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.cell_menu)
        self.table.itemChanged.connect(self.edit_table_number)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self.select_round)
        self.table.cellClicked.connect(self.review_cell)
        layout.addWidget(self.table, 2)
        layout.addWidget(
            QLabel(
                "绿色 · 自动通过 / 人工确认　 黄色 · 候选待复核　 红色 · 缺失需填写"
                "　｜　右键确认，MD / TX 双击编辑"
            )
        )
        footer = QHBoxLayout()
        self.summary = QLabel("尚未导入视频")
        footer.addWidget(self.summary, 1)
        self.missing_hints_button = QPushButton("缺失项提示：开")
        self.missing_hints_button.setCheckable(True)
        self.missing_hints_button.setChecked(True)
        self.missing_hints_button.toggled.connect(self.update_missing_hints)
        footer.addWidget(self.missing_hints_button)
        add = QPushButton("补充缺失轮次")
        add.clicked.connect(self.add_round)
        remove = QPushButton("删除误检轮次")
        remove.clicked.connect(self.remove_round)
        footer.addWidget(add)
        footer.addWidget(remove)
        layout.addLayout(footer)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("等待任务")
        self.progress_bar.setMinimumWidth(220)
        self.progress_bar.setMaximumWidth(300)
        self.progress_bar.setToolTip("按处理阶段和已完成帧/轮次计算进度，不代表预计剩余时间")
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.elapsed_label = QLabel("分析用时：—")
        self.elapsed_label.setToolTip("保留最近一次分析的用时；重新开始分析时清零")
        self.statusBar().addPermanentWidget(self.elapsed_label)
        self.analysis_clock_timer = QTimer(self)
        self.analysis_clock_timer.setInterval(200)
        self.analysis_clock_timer.timeout.connect(self.update_analysis_clock)
        self.cancel_button = QPushButton("取消当前操作")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_work)
        self.statusBar().addPermanentWidget(self.cancel_button)
        self.seek_timer = QTimer(self)
        self.seek_timer.setSingleShot(True)
        self.seek_timer.timeout.connect(lambda: self.show_frame(self.slider.value()))
        for key, delta in (("Left", -1), ("Right", 1), ("Shift+Left", -10), ("Shift+Right", 10)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda d=delta: self.shortcut_step(d))
        self.setStyleSheet("""
            QMainWindow, QWidget { font-family: 'Microsoft YaHei', 'Noto Sans CJK SC', sans-serif;
                font-size: 12px; color: #1e293b; }
            QMainWindow { background: #f3f6fa; }
            QPushButton { padding: 6px 10px; border: 1px solid #cad5df; border-radius: 4px; }
            QPushButton:hover { border-color: #176b87; }
            QGroupBox { border: 1px solid #cad5df; border-radius: 5px; margin-top: 12px; padding-top: 12px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; }
            QTableWidget { background: white; gridline-color: #e2e8f0; }
            QHeaderView::section { background: #e8eef5; padding: 7px; border: none; font-weight: 600; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { padding: 5px; }
            QProgressBar { border: 1px solid #cad5df; border-radius: 4px;
                text-align: center; min-height: 18px; }
            QProgressBar::chunk { background: #69b6c9; border-radius: 3px; }
        """)

    def begin(self, function, completed, *, measure_analysis=False):
        if self.worker and self.worker.isRunning():
            return
        self.seek_timer.stop()
        self.root.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("正在处理")
        self.worker = Worker(function, self)
        self.worker.progress.connect(self.statusBar().showMessage)
        self.worker.percentage.connect(self.show_progress)
        if measure_analysis:
            self.analysis_started = perf_counter()
            self.analysis_elapsed_seconds = 0.0
            self.update_analysis_clock()
            self.analysis_clock_timer.start()
            self.worker.duration.connect(self.finish_analysis_clock)
        self.worker.completed.connect(completed)
        self.worker.failed.connect(self.error)
        self.worker.finished.connect(self.work_finished)
        self.worker.start()

    def update_analysis_clock(self):
        elapsed = (
            perf_counter() - self.analysis_started
            if self.analysis_started is not None
            else self.analysis_elapsed_seconds
        )
        minutes, seconds = divmod(int(elapsed), 60)
        hours, minutes = divmod(minutes, 60)
        self.elapsed_label.setText(f"分析用时：{hours:02d}:{minutes:02d}:{seconds:02d}")

    def finish_analysis_clock(self, elapsed):
        self.analysis_clock_timer.stop()
        self.analysis_started = None
        self.analysis_elapsed_seconds = elapsed
        self.update_analysis_clock()

    def show_progress(self, value):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat("已完成 · %p%" if value == 100 else "%p%")

    def work_finished(self):
        self.root.setEnabled(True)
        self.cancel_button.setEnabled(False)

    def cancel_work(self):
        if self.worker:
            self.worker.requestInterruption()
            self.statusBar().showMessage("正在取消，请等待当前步骤结束…")

    def error(self, message):
        logger.warning("%s", message)
        self.statusBar().showMessage(message)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(max(0, self.progress_bar.value()))
        self.progress_bar.setFormat("已取消 · %p%" if "已取消" in message else "未完成 · %p%")
        if "已取消" not in message:
            QMessageBox.warning(self, "操作未完成", message)

    def may_replace(self):
        return (
            not self.dirty
            or QMessageBox.question(
                self, "尚未保存", "当前任务有未保存的改动，继续会放弃这些改动。是否继续？"
            )
            == QMessageBox.Yes
        )

    def open_video(self):
        if not self.may_replace():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "选择原始视频", "", "视频 (*.mov *.MOV *.mp4 *.mkv *.avi);;全部文件 (*)"
        )
        if path:
            self.begin(
                lambda progress, cancel, percentage: prepare_region_session(path, progress, cancel),
                lambda result: self.accept_session(*result),
            )

    def accept_session(self, session: Session, previews=None, *, saved=False):
        self.editing = True
        self.return_full_view()
        self.region_combo.setCurrentIndex(0)
        if self.frame_reader is not None:
            self.frame_reader.close()
        self.frame_reader = FrameReader(session.video)
        self.region_previews = previews or {}
        self.session = session
        self.canvas.reset_selection()
        self.canvas.rois = session.rois
        maximum = len(session.video.times) - 1
        self.slider.setRange(0, maximum)
        self.frame_input.setRange(-1, maximum)
        for spin in (*self.event_inputs.values(), self.reading_frame):
            spin.setRange(-1, maximum)
        self.method.blockSignals(True)
        self.fps.blockSignals(True)
        self.method.setCurrentIndex(0 if session.method == "frames" else 1)
        self.fps.setValue(session.fps)
        self.method.blockSignals(False)
        self.fps.blockSignals(False)
        self.fps.setEnabled(session.method == "frames")
        self.dirty = not saved
        self.roi_status.setText(f"区域 {len(session.rois)} / 4")
        self.calibration_button.setText(PRESS_MODES[session.press_mode])
        self.refresh_table()
        self.show_frame(
            0, preview=self.region_previews["flash"][1] if self.region_previews else None
        )
        self.details.setPlainText("\n".join(session.analysis_notes + session.video.warnings))
        self.editing = False
        if 0 <= self.table.currentRow() < len(session.rounds):
            self.select_round()
        else:
            self.editing = True
            for spin in (*self.event_inputs.values(), self.reading_frame):
                spin.setValue(-1)
            self.md.clear()
            self.tx.clear()
            self.round_title.setText("选择下方轮次，复核或修正")
            self.editing = False
        self.statusBar().showMessage("任务已就绪。可框选区域、开始分析，或复核现有结果。")

    def show_frame(self, index, *, preview=None):
        if self.session is None or not -1 <= index < len(self.session.video.times):
            return
        self.round_hint.hide()
        self.seek_timer.stop()
        three_frames = (
            self.comparison_field is not None
            and self.video_views.currentWidget() is self.comparison
        )
        pictures = {}
        if index >= 0:
            indices = (
                range(max(0, index - 1), min(len(self.session.video.times), index + 2))
                if three_frames
                else (index,)
            )
            for neighbor in indices:
                try:
                    pictures[neighbor] = (
                        preview
                        if neighbor == index and preview is not None
                        else self.frame_reader.read(neighbor)
                    )
                except (OSError, ValueError) as error:
                    if neighbor == index:
                        self.show_frame(-1)
                        self.error(str(error))
                        return
                    logger.warning("对照帧%d读取失败：%s", neighbor, error)
        image = pictures.get(index)
        self.current_frame = index
        if image is None:
            self.canvas.clear_image("未指定帧\n请用下方导航定位")
        else:
            self.canvas.set_image(image)
        if self.comparison_field is not None:
            item = self.session.rounds[self.table.currentRow()]
            hint = ""
            if self.comparison_field == "press" and item.press.frame is None:
                if item.blink1.frame is None:
                    hint = "press与blink 1均未识别，无法定位查找起点；请用下方导航手工定位。"
                else:
                    start = max(0, item.blink1.frame - 20)
                    hint = (
                        f"press未识别：从blink 1前20帧（原片第{start}帧）开始人工查找；"
                        "仅浏览，未写入press。"
                    )
            if (
                self.comparison_field == "green"
                and item.green_recommendation
                and item.green.state != "manual"
            ):
                recommended = item.green_recommendation.frame
                hint = (
                    f"当前 green 为第{item.green.frame}帧，推荐第{recommended}帧"
                    f"（提前{item.green.frame - recommended}帧）。第二次灭灯后至少两个文字分区的笔画持续变绿。"
                )
            self.review_hint.setText(hint)
            self.review_hint.setVisible(bool(hint))
        if three_frames:
            name = self.comparison_field
            region = REVIEW_REGIONS[name]
            roi = self.session.rois.get(region) if self.comparison.crop_only.isChecked() else None
            label = "MD / TX" if name == "reading" else name.replace("blink", "blink ")
            mode = REGIONS[region] if roi else "全画面"
            self.comparison.title.setText(
                f"第{self.table.currentRow() + 1}轮 · {label} · {mode}"
                + (" · 尚未指定帧" if index < 0 else " · 浏览不修改表格")
            )
            for position, offset in enumerate((-1, 0, 1)):
                neighbor = index + offset
                if index < 0 or not 0 <= neighbor < len(self.session.video.times):
                    message = (
                        "该字段未指定帧\n可用下方导航查找"
                        if index < 0
                        else ("视频起点\n没有上一帧" if offset < 0 else "视频末尾\n没有下一帧")
                    )
                    self.comparison.set_panel(position, None, None, message)
                    continue
                picture = pictures.get(neighbor)
                if picture is None:
                    self.comparison.set_panel(position, neighbor, None, "该帧读取失败")
                    continue
                if roi:
                    if region == "touch":
                        # Preserve the non-planar finger shape instead of rectifying it.
                        height, width = picture.shape[:2]
                        x0, x1 = (
                            math.floor(min(roi[::2]) * (width - 1)),
                            math.ceil(max(roi[::2]) * (width - 1)) + 1,
                        )
                        y0, y1 = (
                            math.floor(min(roi[1::2]) * (height - 1)),
                            math.ceil(max(roi[1::2]) * (height - 1)) + 1,
                        )
                        picture = picture[y0:y1, x0:x1]
                    else:
                        picture = crop(picture, roi)
                self.comparison.set_panel(position, neighbor, picture)
        for widget, value in ((self.slider, max(0, index)), (self.frame_input, index)):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        t = self.session.video.times[index] if index >= 0 else None
        self.time_label.setText(
            "未指定帧" if index < 0 else ("PTS缺失" if t is None else f"PTS {t:.6f} s")
        )

    def return_full_view(self):
        self.comparison_field = None
        self.review_hint.hide()
        self.video_views.setCurrentWidget(self.canvas)
        self.region_visibility.setEnabled(True)
        self.shape_combo.setEnabled(True)

    def change_review_layout(self):
        if self.comparison_field is None:
            return
        three_frames = self.review_layout.currentData() == "comparison"
        self.video_views.setCurrentWidget(self.comparison if three_frames else self.canvas)
        self.region_visibility.setEnabled(not three_frames)
        self.shape_combo.setEnabled(not three_frames)
        self.show_frame(self.current_frame)

    def review_event(self, name, *, preview_frame=None):
        row = self.table.currentRow()
        if self.session is None or not 0 <= row < len(self.session.rounds):
            return
        self.region_combo.setCurrentIndex(0)
        self.comparison_field = name
        three_frames = self.review_layout.currentData() == "comparison"
        self.video_views.setCurrentWidget(self.comparison if three_frames else self.canvas)
        self.region_visibility.setEnabled(not three_frames)
        self.shape_combo.setEnabled(not three_frames)
        item = self.session.rounds[row]
        frame = item.reading_frame if name == "reading" else getattr(item, name).frame
        if name == "press" and frame is None and item.blink1.frame is not None:
            frame = max(0, item.blink1.frame - 20)
        if preview_frame is not None:
            frame = preview_frame
        self.show_frame(frame if frame is not None else -1)

    def slider_changed(self, value):
        self.seek_timer.start(100)

    def step(self, delta):
        if self.session:
            self.show_frame(
                0
                if self.current_frame < 0
                else max(0, min(len(self.session.video.times) - 1, self.current_frame + delta))
            )

    def shortcut_step(self, delta):
        if self.root.isEnabled() and not isinstance(
            QApplication.focusWidget(), (QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox)
        ):
            self.step(delta)

    def region_changed(self, name, roi):
        if self.session:
            if (
                self.session.rounds
                and QMessageBox.question(
                    self, "更改分析区域", "更改区域会清空当前轮次结果，是否继续？"
                )
                != QMessageBox.Yes
            ):
                return
            self.session.rois[name] = roi
            if name == "touch":
                self.session.touch_calibration = None
                self.session.press_mode = "auto"
                self.calibration_button.setText(PRESS_MODES["auto"])
            self.session.rounds = []
            self.session.analysis_notes = []
            if self.session.range_mode == "auto":
                self.session.test_start = self.session.test_end = None
                self.session.range_confirmed = False
            self.dirty = True
            self.roi_status.setText(f"区域 {len(self.session.rois)} / 4")
            self.refresh_table()
            self.statusBar().showMessage("区域已更新；原检测结果已清空，请重新分析")

    def select_region(self):
        name = self.region_combo.currentData()
        self.canvas.set_region(name)
        if name is not None:
            self.return_full_view()
            self.region_visibility.setChecked(True)
            shape = "rectangle" if name == "flash" else "quad"
            self.shape_combo.setCurrentIndex(self.shape_combo.findData(shape))
            if self.session is None:
                return
            if name in self.region_previews:
                frame, image = self.region_previews[name]
                if self.current_frame != frame or self.seek_timer.isActive():
                    self.show_frame(frame, preview=image)
                self.statusBar().showMessage(f"{REGIONS[name]} · 辅助帧 {frame}，可逐帧调整")
            else:
                if self.current_frame < 0:
                    self.show_frame(0)
                self.statusBar().showMessage(f"{REGIONS[name]} · 未找到合适的辅助帧，请手动选帧")

    def start_analysis(self):
        if self.session:
            self.return_full_view()
            self.region_combo.setCurrentIndex(0)
            self.region_visibility.setChecked(False)
            snapshot = (
                self.session
            )  # analyze copies measurement data; diagnostic path also survives failures.
            self.dirty = True
            self.begin(
                lambda progress, cancel, percentage: analyze(
                    snapshot, progress, cancel, percentage
                ),
                self.accept_analysis,
                measure_analysis=True,
            )

    def accept_analysis(self, session):
        self.accept_session(session, self.region_previews)
        if not session.rounds:
            message = "分析已结束，但未识别到有效轮次。请查看检测说明，或导出诊断包。"
            self.round_title.setText("未识别到轮次 · 检测原因如下")
            self.details.setPlainText("\n\n".join(session.analysis_notes + session.video.warnings))
            self.statusBar().showMessage(message)
            QMessageBox.warning(
                self, "没有识别到有效轮次", message + "\n\n" + "\n".join(session.analysis_notes)
            )
            return
        first = next((i for i, r in enumerate(session.rounds) if r.problems()), 0)
        self.table.selectRow(first)
        self.select_round()
        pending = sum(bool(r.problems()) for r in session.rounds)
        self.statusBar().showMessage(
            f"分析完成：检测到{len(session.rounds)}轮（预期10），{pending}轮待确认。"
            "点击下方轮次复核，完成后导出Excel。"
        )

    def timing_changed(self):
        self.fps.setEnabled(self.method.currentData() == "frames")
        if self.session:
            self.session.method = self.method.currentData()
            self.session.fps = self.fps.value()
            self.dirty = True
            self.refresh_table()

    def refresh_table(self):
        if not self.session:
            return
        selected = self.table.currentRow()
        selected_column = self.table.currentColumn()
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.session.rounds))
        totals = []
        for i, item in enumerate(self.session.rounds):
            total = calculate(self.session, item)["end_to_end"]
            if total is not None:
                totals.append(total)
            values = [
                i + 1,
                *[getattr(item, name).frame for name in EVENTS],
                item.md,
                item.tx,
                None if total is None else f"{total:.2f}",
                "；".join(item.problems()) or "已确认",
            ]
            for j, value in enumerate(values):
                cell = QTableWidgetItem("—" if value is None else str(value))
                cell.setFlags(cell.flags() & ~Qt.ItemIsEditable)
                if 1 <= j <= 6:
                    if j <= 4:
                        event = getattr(item, EVENTS[j - 1])
                        state, reason = event.state, event.reason
                    else:
                        name = "md" if j == 5 else "tx"
                        state = item.reading_states.get(name, item.reading_state)
                        reason = item.reading_reason
                        cell.setFlags(cell.flags() | Qt.ItemIsEditable)
                    status = (
                        "missing"
                        if value is None
                        else ("review" if state == "review" else "accepted")
                    )
                    label = {"auto": "自动通过", "manual": "人工确认", "review": "候选待复核"}[
                        state
                    ]
                    cell.setBackground(QColor(CELL_COLORS[status]))
                    cell.setData(
                        Qt.UserRole, ("缺失需填写" if value is None else label) + "\n" + reason
                    )
                    if j == 4 and item.green_recommendation and state != "manual":
                        recommended = item.green_recommendation.frame
                        cell.setText(f"{value} · 荐{recommended}")
                        cell.setData(
                            Qt.UserRole,
                            cell.data(Qt.UserRole)
                            + f"\n笔画变色推荐第{recommended}帧，选中本轮可查看或采用。",
                        )
                elif item.problems():
                    cell.setForeground(QColor("#9a5300"))
                self.table.setItem(i, j, cell)
        self.update_missing_hints()
        if 0 <= selected < len(self.session.rounds):
            self.table.setCurrentCell(selected, max(0, selected_column))
        self.table.blockSignals(False)
        self.update_green_options()
        if not self.session.rounds and self.comparison_field is not None:
            self.return_full_view()
        metadata = rate_summary(self.session.video, self.session.fps)
        if self.session.test_start is not None:
            metadata += f"\n有效片段：{self.session.test_start}～{self.session.test_end}帧（{self.session.range_mode}）"
        else:
            metadata += "\n有效片段：等待识别PRESS HERE起止，可点击“有效片段”手工设置"
        self.metadata.setText(metadata)
        pending = sum(bool(r.problems()) for r in self.session.rounds)
        text = f"{len(self.session.rounds)} / 10轮 · 待确认 {pending}轮"
        if not self.session.range_confirmed:
            text += " · 有效片段待确认"
        if totals:
            text += f" · 端到端均值 {sum(totals) / len(totals):.2f} ms（有值{len(totals)}轮）"
        self.summary.setText(text)

    def update_missing_hints(self):
        enabled = self.missing_hints_button.isChecked()
        self.missing_hints_button.setText("缺失项提示：开" if enabled else "缺失项提示：关")
        if not enabled:
            QToolTip.hideText()
        for row in range(self.table.rowCount()):
            for column in range(self.table.columnCount()):
                cell = self.table.item(row, column)
                text = cell.data(Qt.UserRole) or ""
                if enabled and cell.text() == "—":
                    if column == 1:
                        text = (
                            "未找到对应帧，单击会自动跳转到点击屏幕前方便人工查找。\n"
                            "参考起点为本轮blink 1前20帧；blink 1缺失时请手动定位。"
                        )
                    else:
                        text = (
                            "未自动识别到，单击可以显示对应画面方便人工填写。\n"
                            "没有可用参考帧时，请用下方导航查找。"
                        )
                        if column == 7:
                            text += "\n耗时在补齐press和green后自动计算。"
                cell.setToolTip(text)

    def select_round(self, *, navigate=True):
        index = self.table.currentRow()
        if self.editing or self.session is None or not 0 <= index < len(self.session.rounds):
            return
        self.editing = True
        item = self.session.rounds[index]
        self.round_title.setText(f"第 {index + 1} 轮 · 修改后点击确认")
        self.update_green_options()
        for name, spin in self.event_inputs.items():
            frame = getattr(item, name).frame
            spin.setValue(frame if frame is not None else -1)
        self.md.setText("" if item.md is None else str(item.md))
        self.tx.setText("" if item.tx is None else str(item.tx))
        self.reading_frame.setValue(item.reading_frame if item.reading_frame is not None else -1)
        self.details.setPlainText(
            "\n".join(item.problems()) + "\n\nOCR画面读数：\n" + item.ocr_text
        )
        if not navigate:
            self.editing = False
            return
        frame = next(
            (getattr(item, n).frame for n in EVENTS if getattr(item, n).frame is not None), None
        )
        if self.comparison_field is not None:
            self.review_event(self.comparison_field)
        elif frame is not None:
            self.show_frame(frame)
        else:
            self.show_frame(-1)
        self.editing = False

    def edit_event(self, name, value):
        index = self.table.currentRow()
        if self.editing or self.session is None or not 0 <= index < len(self.session.rounds):
            return
        item = self.session.rounds[index]
        if name in ("green", "blink2"):
            item.green_recommendation = None
        event = getattr(item, name)
        event.frame = value if value >= 0 else None
        event.state, event.reason = "review", "人工修改，待确认"
        self.dirty = True
        self.refresh_table()
        if self.comparison_field == "press" and name in ("press", "blink1"):
            self.show_frame(self.current_frame)
        elif self.comparison_field == "green" and name in ("green", "blink2"):
            self.review_hint.hide()

    def update_green_options(self):
        row = self.table.currentRow()
        item = (
            self.session.rounds[row]
            if self.session and 0 <= row < len(self.session.rounds)
            else None
        )
        rec = item.green_recommendation if item else None
        self.green_options.setVisible(rec is not None)
        if rec is not None:
            adopted = item.green.state == "manual"
            self.preview_green.setVisible(not adopted)
            self.preview_green.setText(f"查看推荐帧 {rec.frame}")
            self.apply_green.setText(
                f"恢复原值 {rec.original.frame}" if adopted else f"采用推荐帧 {rec.frame}"
            )

    def preview_recommended_green(self):
        item = self.session.rounds[self.table.currentRow()]
        self.review_event("green", preview_frame=item.green_recommendation.frame)

    def apply_or_restore_green(self):
        item = self.session.rounds[self.table.currentRow()]
        rec = item.green_recommendation
        if item.green.state == "manual":
            item.green = copy.copy(rec.original)
        else:
            item.green = Event(
                rec.frame, "manual", f"人工采用笔画变色推荐帧，原值为{rec.original.frame}"
            )
        self.dirty = True
        self.refresh_table()
        self.select_round(navigate=False)
        self.review_event("green")

    def edit_number(self, name, text, *, strict=False):
        index = self.table.currentRow()
        if self.editing or self.session is None or not 0 <= index < len(self.session.rounds):
            return
        try:
            value = float(text) if text.strip() else None
            if value is not None and not math.isfinite(value):
                raise ValueError("读数必须为有限数字")
        except ValueError:
            if strict:
                self.statusBar().showMessage("读数请输入有限数字，可为负数或小数，也可留空")
                self.refresh_table()
                return
            value = None
        item = self.session.rounds[index]
        setattr(item, name, value)
        item.set_reading_state(name, "review")
        item.reading_reason = "人工修改，待确认"
        self.dirty = True
        self.refresh_table()
        if strict:
            getattr(self, name).setText("" if value is None else str(value))

    def edit_table_number(self, cell):
        if cell.column() in (5, 6):
            name = "md" if cell.column() == 5 else "tx"
            value = getattr(self.session.rounds[cell.row()], name)
            if cell.text() != ("—" if value is None else str(value)):
                self.edit_number(name, cell.text(), strict=True)

    def cell_menu(self, position):
        cell = self.table.itemAt(position)
        if self.session is None or cell is None or not 1 <= cell.column() <= 6:
            return
        row, column = cell.row(), cell.column()
        item = self.session.rounds[row]
        frame = self.current_frame
        menu = QMenu(self.table)
        if column <= 4:
            candidate = getattr(item, EVENTS[column - 1]).frame
            view = menu.addAction(
                "尚无候选帧" if candidate is None else f"查看候选帧（{candidate}）"
            )
            view.setEnabled(candidate is not None)
            confirm = menu.addAction(
                f"确认当前帧（{frame}）" if frame >= 0 else "确认当前帧（请先显示视频画面）"
            )
            confirm.setEnabled(frame >= 0)
        else:
            view = None
            name = "md" if column == 5 else "tx"
            value = getattr(item, name)
            confirm = menu.addAction(
                f"确认 {name.upper()} 读数（{value:g} ms）"
                if value is not None
                else "确认读数（请先双击填写）"
            )
            confirm.setEnabled(value is not None)
        action = menu.exec(self.table.viewport().mapToGlobal(position))
        if action is None:
            return
        if action == confirm:
            self.confirm_cell(row, column, frame)
        elif action == view:
            self.table.setCurrentCell(row, column)
            self.review_cell(row, column)

    def confirm_cell(self, row, column, frame):
        item = self.session.rounds[row]
        if column <= 4:
            if frame < 0:
                return
            event = getattr(item, EVENTS[column - 1])
            if (
                column == 4
                and item.green_recommendation
                and frame != item.green_recommendation.frame
            ):
                item.green_recommendation = None
            if column == 3 and frame != event.frame:
                item.green_recommendation = None
            event.frame, event.state, event.reason = frame, "manual", "人工确认当前帧"
        else:
            name = "md" if column == 5 else "tx"
            if getattr(item, name) is None:
                return
            item.set_reading_state(name, "manual")
            item.reading_reason = (
                "单项读数已人工复核" if item.reading_state != "review" else "另一读数待复核"
            )
            if item.reading_frame is None and frame >= 0:
                item.reading_frame = frame
        self.dirty = True
        self.table.blockSignals(True)
        self.table.setCurrentCell(row, column)
        self.table.blockSignals(False)
        self.refresh_table()
        self.select_round(navigate=False)
        if column <= 4:
            self.review_cell(row, column)
        elif self.comparison_field is not None:
            self.show_frame(self.current_frame)
        self.statusBar().showMessage("当前单元格已人工确认")

    def edit_reading_frame(self, value):
        index = self.table.currentRow()
        if self.editing or self.session is None or not 0 <= index < len(self.session.rounds):
            return
        item = self.session.rounds[index]
        item.reading_frame = value if value >= 0 else None
        item.reading_states.clear()
        item.reading_state, item.reading_reason = "review", "读数来源已修改，待确认"
        self.dirty = True
        self.refresh_table()

    def review_cell(self, row, column):
        if 1 <= column <= 4:
            self.jump_event(EVENTS[column - 1])
        elif column in (5, 6):
            self.review_event("reading")
        else:
            self.review_event(self.comparison_field or "press")

    def jump_event(self, name):
        self.review_event(name)

    def select_current(self, name):
        if self.current_frame < 0:
            self.statusBar().showMessage("当前没有可用画面，请先选择原片帧")
            return
        if name == "reading":
            self.reading_frame.setValue(self.current_frame)
        else:
            self.event_inputs[name].setValue(self.current_frame)

    def confirm_round(self):
        index = self.table.currentRow()
        if self.session is None or not 0 <= index < len(self.session.rounds):
            return
        try:
            numbers = [
                float(edit.text()) if edit.text().strip() else None for edit in (self.md, self.tx)
            ]
            if any(v is not None and not math.isfinite(v) for v in numbers):
                raise ValueError("MD/TX必须为有限数字")
            frames = [self.event_inputs[n].value() for n in EVENTS]
            if all(f >= 0 for f in frames) and frames != sorted(frames):
                raise ValueError("事件顺序应为 press ≤ blink1 ≤ blink2 ≤ green")
        except ValueError as error:
            self.error(f"输入无效：{error}")
            return
        item = self.session.rounds[index]
        if (
            item.green.state != "manual"
            or item.green.frame != frames[3]
            or item.blink2.frame != frames[2]
        ):
            item.green_recommendation = None
        for name, frame in zip(EVENTS, frames):
            event = getattr(item, name)
            event.frame = frame if frame >= 0 else None
            event.state = "manual" if frame >= 0 else "review"
            event.reason = "人工确认" if frame >= 0 else "尚未选定"
        item.md, item.tx = numbers
        item.reading_states.clear()
        item.reading_frame = self.reading_frame.value() if self.reading_frame.value() >= 0 else None
        item.reading_state = "manual" if all(v is not None for v in numbers) else "review"
        item.reading_reason = "人工确认" if item.reading_state == "manual" else "读数未填全"
        item.issues = []
        self.dirty = True
        self.refresh_table()
        self.select_round()

    def add_round(self):
        if self.session is None:
            return
        rounds = self.session.rounds
        start = self.session.test_start or 0
        if rounds:
            previous = rounds[-1]
            frames = [getattr(previous, name).frame for name in EVENTS]
            frames.extend((previous.reading_frame, previous.last_green_frame))
            last = max((f for f in frames if f is not None), default=None)
            start = max(start, self.current_frame if last is None else last + 1)

        self.return_full_view()
        self.region_combo.setCurrentIndex(0)
        rounds.append(Round())
        self.dirty = True
        self.refresh_table()
        self.table.selectRow(len(rounds) - 1)
        frame = min(start, len(self.session.video.times) - 1)
        self.show_frame(frame)
        if self.current_frame == frame:
            hint = f"已定位到第{len(rounds)}轮的查找起点（原片第{frame}帧），可逐帧观察并录入。"
            if start > frame:
                hint = f"上一轮已到原片末帧（第{frame}帧），请向前查找第{len(rounds)}轮。"
            self.round_hint.setText(hint)
            self.round_hint.show()
            self.round_hint_timer.start(4000)

    def remove_round(self):
        index = self.table.currentRow()
        if self.session and 0 <= index < len(self.session.rounds):
            if QMessageBox.question(self, "删除误检", f"删除第{index + 1}轮？") == QMessageBox.Yes:
                self.session.rounds.pop(index)
                self.dirty = True
                self.refresh_table()
                self.select_round()

    def save_task(self):
        if self.session is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存任务", "analysis.session.json", "任务 (*.json)"
        )
        if path:
            try:
                save_session(self.session, Path(path))
            except OSError as error:
                self.error(str(error))
            else:
                self.dirty = False
                self.statusBar().showMessage("任务已保存")

    def open_task(self):
        if not self.may_replace():
            return
        path, _ = QFileDialog.getOpenFileName(self, "打开任务", "", "任务 (*.json)")
        if path:
            try:
                session = load_session(Path(path))
                if not Path(session.video.path).exists():
                    source, _ = QFileDialog.getOpenFileName(self, "原片已移动，请选择同一个原视频")
                    if not source:
                        return
                    session.video.path = source
                check_source(session.video)
            except (OSError, ValueError, TypeError, KeyError) as error:
                self.error(f"任务无法打开：{error}")
                return
            if self.region_previews and self.session.video == session.video:
                self.accept_session(session, self.region_previews, saved=True)
                return
            self.begin(
                lambda progress, cancel, percentage: prepare_region_session(
                    session.video.path, progress, cancel, session
                ),
                lambda result: self.accept_session(*result, saved=True),
            )

    def export(self):
        if self.session is None:
            return
        parent = QFileDialog.getExistingDirectory(self, "选择结果保存位置")
        if not parent:
            return
        folder = Path(parent) / datetime.now().strftime("PCTS_%Y%m%d_%H%M%S_%f")
        snapshot = copy.deepcopy(self.session)
        self.begin(
            lambda progress, cancel, percentage: export_report(
                snapshot, folder, cancelled=cancel, progress=progress
            ),
            lambda path: self.statusBar().showMessage(f"已导出：{path}"),
        )

    def configure_touch(self):
        if self.session is None:
            return
        if "touch" not in self.session.rois:
            QMessageBox.information(
                self, "指腹设置", "请先框选包含指腹接近和接触位置的触摸观察区域"
            )
            return
        self.return_full_view()
        session = self.session
        roi = session.rois["touch"]
        width, height = session.video.width, session.video.height
        # Use the tracker's original-pixel bounds; no perspective warp for finger geometry.
        x0, y0 = round(min(roi[::2]) * (width - 1)), round(min(roi[1::2]) * (height - 1))
        x1, y1 = round(max(roi[::2]) * (width - 1)) + 1, round(max(roi[1::2]) * (height - 1)) + 1
        dialog = QDialog(self)
        dialog.setWindowTitle("指腹设置 · 自动方向 / 单点取样 / 参考线")
        dialog.resize(680, 660)
        layout = QVBoxLayout(dialog)
        mode = QComboBox()
        mode.addItem("自动：寻找指腹并估计接近方向（推荐）", "auto")
        mode.addItem("单点取样：点一下指腹，自动估计方向", "sample")
        mode.addItem("参考线：沿用旧版位置校准", "reference")
        mode.setCurrentIndex(mode.findData(session.press_mode))
        layout.addWidget(mode)
        instructions = QLabel()
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        frame = QSpinBox()
        frame.setPrefix("参考原帧：")
        frame.setRange(0, len(session.video.times) - 1)
        frame.setKeyboardTracking(False)
        frame.setValue(
            session.touch_calibration.frame
            if session.touch_calibration
            else max(0, self.current_frame)
        )
        layout.addWidget(frame)
        canvas = CalibrationCanvas()
        canvas.setMinimumSize(540, 360)
        layout.addWidget(canvas, 1)
        hint = QLabel()
        hint.setWordWrap(True)
        layout.addWidget(hint)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("保存设置")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        reset = buttons.addButton("重新点选", QDialogButtonBox.ResetRole)
        clear = buttons.addButton("清除取样并恢复自动", QDialogButtonBox.DestructiveRole)
        clear.setEnabled(session.touch_calibration is not None)
        layout.addWidget(buttons)
        reference = None
        proposed = None
        selected = session.touch_calibration
        selected_mode = session.press_mode

        def refresh_preview():
            nonlocal proposed
            proposed = None
            save_button = buttons.button(QDialogButtonBox.Ok)
            save_button.setEnabled(mode.currentData() == "auto")
            if reference is None:
                return
            canvas.set_image(reference[y0:y1, x0:x1])
            if mode.currentData() == "auto":
                hint.setText(
                    "无需点选或画线。分析后点击“查看本轮触摸依据”，检查绿色指腹区域和橙色接近方向。"
                )
                return
            expected = 1 if mode.currentData() == "sample" else 3
            if len(canvas.points) != expected:
                hint.setText(
                    "点击清晰的指腹内部，避开边缘、反光与背景。"
                    if expected == 1
                    else ("点击参考线第一个端点", "点击参考线第二个端点", "点击指腹内部取样点")[
                        len(canvas.points)
                    ]
                )
                return
            points = [
                [(x0 + x * (x1 - x0 - 1)) / (width - 1), (y0 + y * (y1 - y0 - 1)) / (height - 1)]
                for x, y in canvas.points
            ]
            candidate = TouchCalibration(
                frame.value(), [] if expected == 1 else points[0] + points[1], points[-1]
            )
            try:
                validate_touch_calibration(
                    candidate, session.video, roi, require_line=expected == 3
                )
                finger = (
                    MotionTracker(reference, roi, candidate.skin_point)
                    if expected == 1
                    else ContactTracker(reference, roi, candidate)
                )
            except ValueError as error:
                hint.setText(str(error))
                return
            canvas.set_image(motion_preview(finger, {}))
            proposed = candidate
            hint.setText(
                "绿色覆盖应只包含指腹；若覆盖背景，请换帧或换取样点。方向会在分析每轮轨迹后显示。"
            )
            save_button.setEnabled(True)

        def load_reference():
            nonlocal reference
            try:
                reference = read_frame(session.video, frame.value())
            except (OSError, ValueError) as error:
                reference = None
                QMessageBox.warning(dialog, "无法读取参考帧", str(error))
            refresh_preview()

        def change_mode():
            value = mode.currentData()
            canvas.max_points = {"auto": 0, "sample": 1, "reference": 3}[value]
            canvas.points = []
            frame.setEnabled(value != "auto")
            reset.setEnabled(value != "auto")
            instructions.setText(
                {
                    "auto": "默认方式：每轮自动寻找指腹，追踪接近、减速和局部展宽。观察区应包含接近过程。看不清时保留缺失或待复核。",
                    "sample": "选择一帧指腹清晰可见的画面，再点一下指腹内部。不要求该帧已经接触，也不需要画参考线。右键撤销，左键可重新取样。",
                    "reference": "旧版方式：选择明确已接触的一帧，依次点接触参考线两端、指腹内部。参考线穿过实际落点并沿接触表面，不用远处屏幕边框。右键撤销。",
                }[value]
            )
            calibration = session.touch_calibration
            if calibration and value != "auto":
                points = (
                    [calibration.skin_point]
                    if value == "sample"
                    else list(zip(calibration.line[::2], calibration.line[1::2]))
                    + [calibration.skin_point]
                    if calibration.line
                    else []
                )
                canvas.points = [
                    [
                        (x * (width - 1) - x0) / (x1 - x0 - 1),
                        (y * (height - 1) - y0) / (y1 - y0 - 1),
                    ]
                    for x, y in points
                ]
            refresh_preview()

        def save():
            nonlocal selected, selected_mode
            selected_mode = mode.currentData()
            selected = session.touch_calibration if selected_mode == "auto" else proposed
            dialog.accept()

        def clear_saved():
            nonlocal selected, selected_mode
            selected, selected_mode = None, "auto"
            dialog.accept()

        def reset_points():
            canvas.points = []
            refresh_preview()

        canvas.points_changed.connect(refresh_preview)
        frame.valueChanged.connect(load_reference)
        mode.currentIndexChanged.connect(change_mode)
        buttons.accepted.connect(save)
        buttons.rejected.connect(dialog.reject)
        reset.clicked.connect(reset_points)
        clear.clicked.connect(clear_saved)
        load_reference()
        change_mode()
        if dialog.exec() != QDialog.Accepted or (
            selected == session.touch_calibration and selected_mode == session.press_mode
        ):
            return
        session.touch_calibration, session.press_mode = selected, selected_mode
        for item in session.rounds:
            if item.press.state != "manual":
                item.press = Event(reason="指腹设置已更改，请重新分析")
        self.calibration_button.setText(PRESS_MODES[selected_mode])
        self.region_combo.setCurrentIndex(0)
        self.dirty = True
        self.refresh_table()
        self.select_round()
        if selected and selected_mode != "auto":
            self.show_frame(selected.frame)
        self.statusBar().showMessage("指腹设置已保存，请重新分析")

    def show_touch_evidence(self):
        row = self.table.currentRow()
        if self.session is None or row < 0:
            return
        if not self.session.diagnostic_dir:
            QMessageBox.information(self, "触摸依据", "尚无本次分析的触摸记录，请先分析视频。")
            return
        directory = log_directory() / Path(self.session.diagnostic_dir).name
        report_path = directory / "touch-detection.json"
        if not report_path.is_file():
            QMessageBox.information(self, "触摸依据", "尚无本次分析的触摸记录，请先分析视频。")
            return
        reports = json.loads(report_path.read_text(encoding="utf-8"))
        blink1 = self.session.rounds[row].blink1.frame
        report = next(
            (r for r in reports if blink1 is not None and r.get("blink1") == blink1), None
        )
        if report is None:
            QMessageBox.information(
                self, "触摸依据", "本轮没有对应的新版本记录，或轮次已被修改；请重新分析。"
            )
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"第{row + 1}轮 · 自动分析时的触摸依据")
        dialog.resize(600, 500)
        layout = QVBoxLayout(dialog)
        target = report.get("target", {})
        text = QLabel(
            f"指腹参考帧：{target.get('reference_frame', '未取样')}\n"
            "绿色：选中的指腹区域；橙色箭头：本轮估计的接近方向。\n"
            "此图用于核对目标与方向，不表示这一帧已接触。\n" + report["event"]["reason"]
        )
        text.setWordWrap(True)
        layout.addWidget(text)
        preview = QLabel()
        preview.setAlignment(Qt.AlignCenter)
        path = directory / f"touch-preview-{report['round']:02d}.png"
        if path.is_file():
            preview.setPixmap(
                QPixmap(str(path)).scaled(520, 350, Qt.KeepAspectRatio, Qt.FastTransformation)
            )
        else:
            preview.setText("没有自动轨迹预览；参考线模式可在指腹设置中查看取样。")
        layout.addWidget(preview, 1)
        close = QDialogButtonBox(QDialogButtonBox.Close)
        close.rejected.connect(dialog.reject)
        layout.addWidget(close)
        dialog.exec()

    def configure_range(self):
        if self.session is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("有效测试片段")
        layout = QFormLayout(dialog)
        mode = QComboBox()
        mode.addItem("自动：PRESS HERE出现至消失", "auto")
        mode.addItem("手工指定原片帧范围", "manual")
        mode.setCurrentIndex(0 if self.session.range_mode == "auto" else 1)
        layout.addRow("方式", mode)
        start, end = QSpinBox(), QSpinBox()
        for spin in (start, end):
            spin.setRange(0, len(self.session.video.times) - 1)
        start.setValue(self.session.test_start or 0)
        end.setValue(
            self.session.test_end
            if self.session.test_end is not None
            else len(self.session.video.times) - 1
        )
        end.setMinimum(start.value())
        start.valueChanged.connect(end.setMinimum)
        layout.addRow("起帧（包含）", start)
        layout.addRow("止帧（包含）", end)
        for spin in (start, end):
            spin.setEnabled(mode.currentData() == "manual")
        mode.currentIndexChanged.connect(lambda: start.setEnabled(mode.currentData() == "manual"))
        mode.currentIndexChanged.connect(lambda: end.setEnabled(mode.currentData() == "manual"))
        info = QLabel(
            "连续4次检测到 PRESS HERE 消失且文字选区中央空白时，自动确定结束帧。\n"
            "自动边界有疑问时，可在原片查看帧号后手工确认范围。"
        )
        info.setWordWrap(True)
        layout.addRow(info)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec() != QDialog.Accepted:
            return
        selected = (start.value(), end.value()) if mode.currentData() == "manual" else (None, None)
        if mode.currentData() == self.session.range_mode and (
            mode.currentData() == "auto"
            or selected == (self.session.test_start, self.session.test_end)
        ):
            if mode.currentData() == "manual":
                self.session.range_confirmed = True
                self.dirty = True
                self.refresh_table()
            return
        changed = selected != (self.session.test_start, self.session.test_end)
        if (
            changed
            and self.session.rounds
            and QMessageBox.question(
                self, "重新分析", "有效片段改变后需要重新分析，清空当前结果并继续？"
            )
            != QMessageBox.Yes
        ):
            return
        self.session.range_mode = mode.currentData()
        self.session.test_start, self.session.test_end = selected
        self.session.range_confirmed = self.session.range_mode == "manual"
        if changed:
            self.session.rounds = []
            self.session.analysis_notes = []
        self.dirty = True
        self.refresh_table()
        self.statusBar().showMessage("有效片段设置已更新；范围变化后请重新分析")

    def show_analysis_notes(self):
        text = (
            "\n\n".join(self.session.analysis_notes + self.session.video.warnings)
            if self.session
            else "尚未分析"
        )
        QMessageBox.information(self, "检测说明", text + f"\n\n诊断日志位置：{log_directory()}")

    def export_debug(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "导出诊断包", "PCTS-diagnostics.zip", "诊断包 (*.zip)"
        )
        if path:
            try:
                export_diagnostics(self.session, Path(path))
            except OSError as error:
                self.error(str(error))
            else:
                self.statusBar().showMessage(
                    f"诊断包已导出：{path}（含日志和识别数据，不含原视频）"
                )

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.cancel_work()
            event.ignore()
            return
        if self.may_replace():
            if self.frame_reader is not None:
                self.frame_reader.close()
            event.accept()
        else:
            event.ignore()
