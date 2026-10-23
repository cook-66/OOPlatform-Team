#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""ODPlatform Desktop demo (PySide6).

验证 D8 接缝设计:
- 不修改 inference/ 任何代码
- 通过 QtSignalSink / InferHooks / CancelToken 三个接缝跟 InferService 集成
- UI 不冻 (推理跑在 QThread)
- Stop < 2s 响应 (CancelToken 4 个查询点)

运行:
    pip install PySide6
    python apps/desktop/main.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

# 让 desktop app 能 import platform 包 (monorepo 跑法)
THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1] / "platform" / "src"))

from infer_worker import InferWorker
from qt_sink import QtSignalSink

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("desktop")


# ============================================================================
# logging → QPlainTextEdit 桥
# ============================================================================
class _LogBridge(QObject):
    """logging.Handler 在任意线程 emit, 经 Qt 信号排队到主线程追加文本."""
    message = Signal(str)


class QtLogHandler(logging.Handler):
    """把 logging 记录转发进 Qt 日志框 (线程安全)."""

    def __init__(self, bridge: _LogBridge) -> None:
        super().__init__()
        self._bridge = bridge
        self.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S",
        ))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._bridge.message.emit(self.format(record))
        except Exception:
            self.handleError(record)


class MainWindow(QMainWindow):
    """最小桌面 demo:
      - 模型 / 源 两个文本框 + Browse 按钮
      - Start / Stop 按钮
      - 实时画面 QLabel
      - 进度条 + 状态栏 + 日志框
    """

    request_start = Signal(str, str)    # model_path, source

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ODPlatform Desktop Demo")
        self.resize(1280, 800)
        self._running = False
        self._build_ui()
        self._install_log_handler()
        self._build_worker()

    # ---- UI ----
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        v = QVBoxLayout(central)

        # 模型行
        row_model = QHBoxLayout()
        row_model.addWidget(QLabel("Model:"))
        self.le_model = QLineEdit("yolo11n.pt")
        self.le_model.setPlaceholderText("模型文件名或路径, 如 yolo11n.pt / best.pt")
        row_model.addWidget(self.le_model, 1)
        btn_browse_model = QPushButton("Browse...")
        btn_browse_model.clicked.connect(self._browse_model)
        row_model.addWidget(btn_browse_model)
        v.addLayout(row_model)

        # 源行
        row_src = QHBoxLayout()
        row_src.addWidget(QLabel("Source:"))
        self.le_source = QLineEdit("0")
        self.le_source.setPlaceholderText("摄像头号 (0/1...) 或 视频文件路径 或 图片目录")
        row_src.addWidget(self.le_source, 1)
        btn_browse_src = QPushButton("File...")
        btn_browse_src.clicked.connect(self._browse_source_file)
        row_src.addWidget(btn_browse_src)
        btn_browse_dir = QPushButton("Dir...")
        btn_browse_dir.clicked.connect(self._browse_source_dir)
        row_src.addWidget(btn_browse_dir)
        v.addLayout(row_src)

        # Start / Stop 行
        row_btn = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        row_btn.addWidget(self.btn_start)
        row_btn.addWidget(self.btn_stop)
        v.addLayout(row_btn)

        # 进度条
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)           # 启动时 indeterminate
        self.progress.setVisible(False)
        v.addWidget(self.progress)

        # 实时画面
        self.lbl_view = QLabel("等待启动...")
        self.lbl_view.setMinimumSize(640, 360)
        self.lbl_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_view.setStyleSheet("background:#111; color:#888;")
        v.addWidget(self.lbl_view, 3)

        # 日志框
        self.te_log = QPlainTextEdit()
        self.te_log.setReadOnly(True)
        self.te_log.setMaximumBlockCount(5000)
        self.te_log.setMinimumHeight(140)
        v.addWidget(self.te_log, 1)

        # 状态栏: 左=状态, 右=帧率
        self.lbl_status = QLabel("就绪")
        self.lbl_fps = QLabel("FPS -")
        self.statusBar().addWidget(self.lbl_status, 1)
        self.statusBar().addPermanentWidget(self.lbl_fps)

    def _install_log_handler(self) -> None:
        """把 root logger 接到日志框 (所有线程的日志都进 UI)."""
        self._log_bridge = _LogBridge(self)
        self._log_bridge.message.connect(self.te_log.appendPlainText)
        self._log_handler = QtLogHandler(self._log_bridge)
        logging.getLogger().addHandler(self._log_handler)

    def _build_worker(self) -> None:
        # sink 必须在主线程构造 (内部有 Qt Signal)
        self.sink = QtSignalSink()
        self.sink.frame_ready.connect(self._on_frame)

        self.worker = InferWorker()
        self.worker.set_sink(self.sink)

        self.thread = QThread(self)
        self.worker.moveToThread(self.thread)
        self.request_start.connect(self.worker.start_infer)
        self.worker.progress_changed.connect(self._on_progress)
        self.worker.inference_finished.connect(self._on_finished)
        self.worker.inference_failed.connect(self._on_failed)
        self.thread.start()

    # ---- 按钮/文件选择 ----
    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型", "", "PyTorch 模型 (*.pt *.pth);;所有文件 (*)",
        )
        if path:
            self.le_model.setText(path)

    def _browse_source_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频/图片",
            "", "媒体文件 (*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp);;所有文件 (*)",
        )
        if path:
            self.le_source.setText(path)

    def _browse_source_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if path:
            self.le_source.setText(path)

    # ---- Start / Stop ----
    def _on_start(self) -> None:
        if self._running:
            return
        model = self.le_model.text().strip()
        source = self.le_source.text().strip()
        if not model or not source:
            self._append_log("启动失败: 模型和输入源都不能为空")
            self.lbl_status.setText("启动失败: 模型/源为空")
            return

        self._running = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.lbl_status.setText("启动中...")
        self.lbl_fps.setText("FPS -")
        self._append_log(f"--- 启动推理: model={model}, source={source} ---")
        self.request_start.emit(model, source)

    def _on_stop(self) -> None:
        if not self._running:
            return
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("停止中...")
        self._append_log("--- 请求停止 (CancelToken) ---")
        # ★ 直接调用而不是排队信号: start_infer 正阻塞 worker 线程的事件循环,
        #   排队的 cancel 槽要等推理结束才会执行; cancel() 只 set 一个 Event, 线程安全.
        self.worker.cancel()

    # ---- worker 信号槽 ----
    def _on_progress(self, frame_idx: int, total: int, fps_loop: float) -> None:
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(min(frame_idx, total))
            progress_text = f"{frame_idx}/{total}"
        else:
            self.progress.setRange(0, 0)       # 摄像头/流: 无总帧数
            progress_text = f"{frame_idx} 帧"
        self.lbl_status.setText(f"运行中 | {progress_text} | loop {fps_loop:.1f} FPS")
        self.lbl_fps.setText(f"FPS {fps_loop:.1f}")

    def _on_frame(self, annotated) -> None:
        if not isinstance(annotated, np.ndarray) or annotated.ndim != 3 or annotated.size == 0:
            return
        # BGR → RGB (视图即可, QImage.copy() 会拷走)
        rgb = np.ascontiguousarray(annotated[:, :, ::-1])
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg).scaled(
            self.lbl_view.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.lbl_view.setPixmap(pix)

    def _on_finished(self, result) -> None:
        self._finish_run()
        stats = result.stats if isinstance(result.stats, dict) else {}
        frames = stats.get("frames", 0)
        dets = stats.get("detections", 0)
        avg_fps = stats.get("avg_fps", 0.0)
        text = f"推理结束 | success={result.success} | 帧数 {frames} | 检测 {dets} | 平均 {avg_fps} FPS"
        if result.output_dir:
            text += f" | 输出 {result.output_dir}"
        if result.error:
            text += f" | error={result.error}"
        self._append_log(text)
        self.lbl_status.setText(text)

    def _on_failed(self, message: str) -> None:
        self._finish_run()
        self._append_log(f"推理失败: {message}")
        self.lbl_status.setText(f"推理失败: {message}")

    def _finish_run(self) -> None:
        self._running = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress.setVisible(False)
        self.progress.setRange(0, 0)

    def _append_log(self, text: str) -> None:
        self.te_log.appendPlainText(text)

    # ---- 退出 ----
    def closeEvent(self, event) -> None:
        if self._running:
            self.worker.cancel()
        self.thread.quit()
        self.thread.wait(3000)
        logging.getLogger().removeHandler(self._log_handler)
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
