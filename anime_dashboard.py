import sys
import time
import csv
import math
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import pyqtgraph as pg

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFrame,
    QFileDialog, QMessageBox
)

from face_utils import average_ear, gaze_offset, head_pose
from cognitive_load import CognitiveLoadEstimator


# ---------- Theme ----------
BG = "#090B18"
PANEL = "#14182B"
CYAN = "#42F5E9"
PURPLE = "#B66CFF"
PINK = "#FF4FA3"
GREEN = "#4DFFA6"
YELLOW = "#FFD166"
WHITE = "#F4F6FF"
MUTED = "#9CA8C7"


def make_label(text, size=12, color=WHITE, bold=False):
    label = QLabel(text)
    label.setFont(QFont("Segoe UI", size, QFont.Weight.Bold if bold else QFont.Weight.Normal))
    label.setStyleSheet(f"color: {color}; background: transparent;")
    return label


def make_card(title, value, color):
    card = QFrame()
    card.setStyleSheet(
        f"QFrame {{background:{PANEL}; border:1px solid #303653;"
        "border-radius:14px;}"
    )
    layout = QVBoxLayout(card)
    layout.setContentsMargins(14, 12, 14, 12)
    layout.addWidget(make_label(title.upper(), 9, MUTED, True))
    number = make_label(value, 21, color, True)
    layout.addWidget(number)
    layout.addStretch()
    return card, number


class ScoreGauge(QWidget):
    def __init__(self):
        super().__init__()
        self.score = 0
        self.phase = 0
        self.setMinimumHeight(175)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(40)

    def animate(self):
        self.phase += 4
        self.update()

    def set_score(self, score):
        self.score = max(0, min(100, int(score)))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        side = min(self.width(), self.height()) - 30
        rect = self.rect().adjusted(
            (self.width() - side) // 2,
            (self.height() - side) // 2,
            -(self.width() - side) // 2,
            -(self.height() - side) // 2
        )

        color = QColor(
            GREEN if self.score < 35 else
            YELLOW if self.score < 70 else PINK
        )

        p.setPen(QPen(QColor("#292E49"), 12))
        p.drawArc(rect, 0, 360 * 16)

        p.setPen(QPen(color, 12, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(rect, 90 * 16, -int(self.score * 3.6 * 16))

        p.setPen(QColor(WHITE))
        p.setFont(QFont("Segoe UI", 27, QFont.Weight.Bold))
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(self.score))

        p.setFont(QFont("Segoe UI", 9))
        p.setPen(QColor(MUTED))
        p.drawText(
            rect.adjusted(0, 48, 0, 0),
            Qt.AlignmentFlag.AlignCenter, "LOAD SCORE / 100"
        )


class AnimeDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NEURO VISION | Cognitive Load AI")
        self.resize(1450, 900)
        self.setMinimumSize(1000, 700)

        self.camera = cv2.VideoCapture(0)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.estimator = CognitiveLoadEstimator(window_seconds=15)
        self.closed_frames = 0
        self.total_blinks = 0
        self.started = time.time()
        self.face_found = False
        self.current_score = 0
        self.history_time = []
        self.history_load = []
        self.history_blink = []
        self.last_graph_update = 0
        self.last_result = {}

        self.build_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def build_ui(self):
        central = QWidget()
        central.setStyleSheet(f"background:{BG}; color:{WHITE};")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(18, 15, 18, 15)
        root.setSpacing(12)

        header = QHBoxLayout()
        title = make_label("✦ NEURO VISION", 23, CYAN, True)
        subtitle = make_label("ANIME AI • COGNITIVE MONITORING", 10, MUTED)
        heading = QVBoxLayout()
        heading.addWidget(title)
        heading.addWidget(subtitle)
        header.addLayout(heading)
        header.addStretch()

        self.status = make_label("● INITIALIZING CAMERA", 11, YELLOW, True)
        header.addWidget(self.status)
        root.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(14)
        root.addLayout(body, 5)

        # Webcam panel
        camera_panel = QFrame()
        camera_panel.setStyleSheet(
            f"background:{PANEL};border:1px solid #303653;border-radius:14px;"
        )
        cam_layout = QVBoxLayout(camera_panel)
        cam_layout.addWidget(make_label("◈ LIVE FACE SCANNER", 12, CYAN, True))

        self.video = QLabel("Opening webcam...")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(450, 320)
        self.video.setStyleSheet(
            "background:#050712;border:1px solid #283653;border-radius:10px;"
        )
        cam_layout.addWidget(self.video, 1)

        self.face_status = make_label("Waiting for face...", 11, MUTED)
        cam_layout.addWidget(self.face_status)
        body.addWidget(camera_panel, 6)

        # Right dashboard
        right = QVBoxLayout()
        right.setSpacing(10)
        body.addLayout(right, 5)

        gauge_card = QFrame()
        gauge_card.setStyleSheet(
            f"background:{PANEL};border:1px solid #303653;border-radius:14px;"
        )
        gauge_layout = QVBoxLayout(gauge_card)
        gauge_layout.addWidget(make_label("◉ COGNITIVE LOAD", 12, PURPLE, True))
        self.gauge = ScoreGauge()
        gauge_layout.addWidget(self.gauge)
        self.level = make_label("WAITING FOR DATA", 13, GREEN, True)
        self.level.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gauge_layout.addWidget(self.level)
        right.addWidget(gauge_card, 3)

        grid = QGridLayout()
        grid.setSpacing(9)
        right.addLayout(grid, 2)

        card_specs = [
            ("Total blinks", "0", CYAN),
            ("Blink / min", "0.0", PURPLE),
            ("Gaze variance", "0.00", GREEN),
            ("Head movement", "0.00", YELLOW),
        ]
        self.values = {}
        for i, (name, initial, color) in enumerate(card_specs):
            card, value = make_card(name, initial, color)
            grid.addWidget(card, i // 2, i % 2)
            self.values[name] = value

        # Graphs
        graph_row = QHBoxLayout()
        root.addLayout(graph_row, 3)

        self.load_graph = pg.PlotWidget(title="COGNITIVE LOAD HISTORY")
        self.blink_graph = pg.PlotWidget(title="BLINK RATE HISTORY")

        for graph in (self.load_graph, self.blink_graph):
            graph.setBackground(PANEL)
            graph.showGrid(x=True, y=True, alpha=0.2)
            graph.getAxis("left").setTextPen(MUTED)
            graph.getAxis("bottom").setTextPen(MUTED)
            graph.setLabel("bottom", "Session time (seconds)", color=MUTED)
            graph.setMinimumHeight(170)

        self.load_graph.setLabel("left", "Score (0–100)", color=MUTED)
        self.blink_graph.setLabel("left", "Blinks per minute", color=MUTED)

        self.load_line = self.load_graph.plot(pen=pg.mkPen(CYAN, width=2))
        self.blink_line = self.blink_graph.plot(pen=pg.mkPen(PURPLE, width=2))
        graph_row.addWidget(self.load_graph)
        graph_row.addWidget(self.blink_graph)

        # Bottom controls
        controls = QHBoxLayout()
        root.addLayout(controls)

        self.session_label = make_label("SESSION 00:00", 10, MUTED)
        controls.addWidget(self.session_label)
        controls.addStretch()

        export_btn = QPushButton("EXPORT CSV")
        reset_btn = QPushButton("RESET SESSION")
        for button in (export_btn, reset_btn):
            button.setStyleSheet(
                f"QPushButton{{background:{PANEL};color:{CYAN};"
                "border:1px solid #3D5475;border-radius:8px;"
                "padding:10px 18px;font-weight:bold;}"
                "QPushButton:hover{background:#242C48;}"
            )
        export_btn.clicked.connect(self.export_csv)
        reset_btn.clicked.connect(self.reset_session)
        controls.addWidget(export_btn)
        controls.addWidget(reset_btn)

    def update_frame(self):
        if not self.camera.isOpened():
            self.status.setText("● CAMERA UNAVAILABLE")
            self.status.setStyleSheet(f"color:{PINK};")
            return

        ok, frame = self.camera.read()
        if not ok:
            return

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        self.face_found = bool(result.multi_face_landmarks)

        if self.face_found:
            landmarks = result.multi_face_landmarks[0].landmark

            # Draw MediaPipe face landmarks.
            for lm in landmarks:
                px, py = int(lm.x * w), int(lm.y * h)
                if 0 <= px < w and 0 <= py < h:
                    cv2.circle(frame, (px, py), 1, (235, 110, 255), -1)

            ear = average_ear(landmarks, w, h)
            self.estimator.add_ear_sample(ear)

            if ear < 0.21:
                self.closed_frames += 1
            else:
                if self.closed_frames >= 2:
                    self.total_blinks += 1
                    self.estimator.add_blink()
                self.closed_frames = 0

            gaze = gaze_offset(landmarks, w, h)
            self.estimator.add_gaze_sample(gaze)

            pose = head_pose(landmarks, w, h)
            if pose is not None:
                pitch, yaw, roll = pose
                self.estimator.add_head_pose(pitch, yaw, roll)

            data = self.estimator.compute_load()
            self.last_result = data
            self.current_score = float(data["score"])
            self.gauge.set_score(self.current_score)

            level = (
                "LOW LOAD" if self.current_score < 35 else
                "MEDIUM LOAD" if self.current_score < 70 else
                "HIGH LOAD"
            )
            color = (
                GREEN if self.current_score < 35 else
                YELLOW if self.current_score < 70 else PINK
            )
            self.level.setText(level)
            self.level.setStyleSheet(f"color:{color};font-weight:bold;")
            self.status.setText("● FACE TRACKING ACTIVE")
            self.status.setStyleSheet(f"color:{GREEN};")
            self.face_status.setText("✦ Face detected • Tracking active")
            self.face_status.setStyleSheet(f"color:{CYAN};")

            self.values["Total blinks"].setText(str(self.total_blinks))
            self.values["Blink / min"].setText(
                f'{float(data["blink_rate"]):.1f}'
            )
            self.values["Gaze variance"].setText(
                f'{float(data["gaze_variance"]):.2f}'
            )
            self.values["Head movement"].setText(
                f'{float(data["head_movement"]):.2f}'
            )

            # Animated scanning line.
            scan_y = int((time.time() * 100) % h)
            cv2.line(frame, (0, scan_y), (w, scan_y), (255, 220, 60), 1)
        else:
            self.status.setText("● SEARCHING FOR FACE")
            self.status.setStyleSheet(f"color:{YELLOW};")
            self.face_status.setText("Move into the camera view")
            self.face_status.setStyleSheet(f"color:{YELLOW};")

        # Send webcam image to the interface.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(
            rgb_frame.data, w, h, 3 * w,
            QImage.Format.Format_RGB888
        ).copy()
        pixmap = QPixmap.fromImage(image)
        self.video.setPixmap(
            pixmap.scaled(
                self.video.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

        now = time.time()
        elapsed = int(now - self.started)
        mins, secs = divmod(elapsed, 60)
        self.session_label.setText(f"SESSION {mins:02d}:{secs:02d}")

        # Keep a time series for live graphs.
        if self.face_found and now - self.last_graph_update >= 0.5:
            self.history_time.append(now - self.started)
            self.history_load.append(self.current_score)
            self.history_blink.append(
                float(self.last_result.get("blink_rate", 0))
            )
            self.last_graph_update = now

            self.load_line.setData(self.history_time, self.history_load)
            self.blink_line.setData(self.history_time, self.history_blink)

    def export_csv(self):
        if not self.history_time:
            QMessageBox.information(
                self, "No data yet",
                "Keep the webcam running until some measurements are collected."
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export session data",
            f"cognitive_session_{datetime.now():%Y%m%d_%H%M%S}.csv",
            "CSV files (*.csv)"
        )
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Elapsed seconds", "Cognitive load score",
                    "Blink rate per minute"
                ])
                writer.writerows(zip(
                    self.history_time,
                    self.history_load,
                    self.history_blink
                ))
            QMessageBox.information(self, "Export complete", "CSV report saved.")
        except OSError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def reset_session(self):
        self.estimator = CognitiveLoadEstimator(window_seconds=15)
        self.closed_frames = 0
        self.total_blinks = 0
        self.started = time.time()
        self.current_score = 0
        self.history_time.clear()
        self.history_load.clear()
        self.history_blink.clear()
        self.last_graph_update = 0
        self.last_result = {}

        self.gauge.set_score(0)
        self.level.setText("WAITING FOR DATA")
        for name, value in self.values.items():
            value.setText("0.0" if name != "Total blinks" else "0")
        self.load_line.setData([], [])
        self.blink_line.setData([], [])

    def closeEvent(self, event):
        self.timer.stop()
        self.camera.release()
        self.face_mesh.close()
        cv2.destroyAllWindows()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = AnimeDashboard()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()