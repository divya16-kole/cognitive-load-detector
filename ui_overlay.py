import time
from collections import deque

import cv2
import numpy as np

# OpenCV uses BGR colors, not RGB.
BG = (20, 16, 30)
WHITE = (245, 245, 255)
MUTED = (155, 165, 185)
CYAN = (255, 220, 60)
PURPLE = (220, 100, 255)
GREEN = (90, 230, 120)
YELLOW = (60, 210, 255)
RED = (80, 80, 255)


def put(frame, text, pos, scale=0.6, color=WHITE, thickness=1):
    cv2.putText(
        frame, str(text), pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness, cv2.LINE_AA
    )


class Dashboard:
    def __init__(self, window_seconds=60):
        self.window_seconds = window_seconds
        self.started = time.time()
        self.last_update = time.time()

        self.score = 0.0
        self.blink_rate = 0.0
        self.total_blinks = 0
        self.gaze_variance = 0.0
        self.head_movement = 0.0

        self.history = deque()
        self.last_graph_sample = 0.0
        self.pulse = 0.0

    def update(
        self, score, blink_rate, total_blinks,
        gaze_variance, head_movement
    ):
        now = time.time()

        self.score = float(np.clip(score, 0, 100))
        self.blink_rate = max(0.0, float(blink_rate))
        self.total_blinks = int(total_blinks)
        self.gaze_variance = max(0.0, float(gaze_variance))
        self.head_movement = max(0.0, float(head_movement))

        # Sample graph approximately twice per second.
        if now - self.last_graph_sample >= 0.5:
            self.history.append(
                (now, self.score, self.blink_rate)
            )
            self.last_graph_sample = now

        cutoff = now - self.window_seconds
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

        self.last_update = now

    def _level(self):
        if self.score < 35:
            return "LOW", GREEN
        elif self.score < 70:
            return "MEDIUM", YELLOW
        return "HIGH", RED

    def _panel(self, frame, x1, y1, x2, y2, alpha=0.86):
        overlay = frame.copy()
        cv2.rectangle(
            overlay, (x1, y1), (x2, y2), BG, -1
        )
        cv2.addWeighted(
            overlay, alpha, frame, 1 - alpha, 0, frame
        )
        cv2.rectangle(
            frame, (x1, y1), (x2, y2),
            (75, 65, 100), 1, cv2.LINE_AA
        )

    def _draw_graph(self, frame, x, y, width, height):
        self._panel(frame, x, y, x + width, y + height)

        put(
            frame, "LIVE LOAD HISTORY",
            (x + 12, y + 22), 0.48, CYAN, 1
        )

        left = x + 35
        right = x + width - 12
        top = y + 40
        bottom = y + height - 20

        if right <= left or bottom <= top:
            return

        # Graph grid.
        for fraction in (0, 0.5, 1):
            gy = int(bottom - fraction * (bottom - top))
            cv2.line(
                frame, (left, gy), (right, gy),
                (65, 58, 78), 1, cv2.LINE_AA
            )

        put(frame, "100", (x + 3, top + 4), 0.3, MUTED)
        put(frame, "50", (x + 9, (top + bottom) // 2),
            0.3, MUTED)
        put(frame, "0", (x + 15, bottom), 0.3, MUTED)

        if len(self.history) < 2:
            put(
                frame, "Collecting live data...",
                (left + 8, (top + bottom) // 2),
                0.42, MUTED
            )
            return

        now = time.time()
        points_load = []
        points_blink = []

        for timestamp, score, blink in self.history:
            age = now - timestamp
            px = right - int(
                min(age / max(self.window_seconds, 1), 1)
                * (right - left)
            )

            py_load = bottom - int(
                np.clip(score, 0, 100) / 100 * (bottom - top)
            )

            # Blink rate uses its own display scale for visibility.
            blink_display = np.clip(blink, 0, 60) / 60
            py_blink = bottom - int(
                blink_display * (bottom - top)
            )

            points_load.append((px, py_load))
            points_blink.append((px, py_blink))

        for points, color in (
            (points_load, CYAN),
            (points_blink, PURPLE),
        ):
            for i in range(1, len(points)):
                cv2.line(
                    frame, points[i - 1], points[i],
                    color, 2, cv2.LINE_AA
                )

        # Legend.
        cv2.line(
            frame, (left, y + 31),
            (left + 15, y + 31), CYAN, 2
        )
        put(frame, "Load", (left + 20, y + 35), 0.32, WHITE)

        cv2.line(
            frame, (left + 82, y + 31),
            (left + 97, y + 31), PURPLE, 2
        )
        put(
            frame, "Blink rate (scaled)",
            (left + 102, y + 35), 0.32, WHITE
        )

    def draw(self, frame):
        h, w = frame.shape[:2]
        if w < 200 or h < 200:
            return frame

        now = time.time()
        self.pulse = (np.sin(now * 3) + 1) / 2

        # Scale the interface for different webcam resolutions.
        scale = w / 640.0
        def S(value):
            return max(1, int(value * scale))

        margin = S(12)
        gap = S(8)
        header_h = S(43)
        card_h = S(82)
        graph_h = S(155)

        # Header.
        self._panel(
            frame, margin, margin,
            w - margin, margin + header_h
        )

        put(
            frame, "COGNITIVE LOAD DETECTOR",
            (margin + S(12), margin + S(19)),
            0.58 * scale, CYAN, S(2)
        )
        put(
            frame, "AI  /  LIVE MONITOR",
            (margin + S(13), margin + S(35)),
            0.34 * scale, MUTED
        )

        # Animated status indicator.
        dot_x = w - margin - S(17)
        dot_y = margin + header_h // 2
        dot_color = (
            GREEN if self.pulse > 0.25 else (40, 150, 75)
        )
        cv2.circle(frame, (dot_x, dot_y), S(5), dot_color, -1)
        put(
            frame, "LIVE",
            (dot_x - S(53), dot_y + S(4)),
            0.38 * scale, GREEN
        )

        # Four metric cards.
        card_y = margin + header_h + gap
        usable_w = w - 2 * margin
        card_w = (usable_w - 3 * gap) // 4

        level, level_color = self._level()
        cards = [
            ("LOAD SCORE", f"{self.score:.0f} / 100",
             level_color),
            ("BLINKS", str(self.total_blinks), CYAN),
            ("BLINK / MIN", f"{self.blink_rate:.1f}", PURPLE),
            ("HEAD MOVEMENT", f"{self.head_movement:.2f}", YELLOW),
        ]

        for i, (title, value, color) in enumerate(cards):
            x1 = margin + i * (card_w + gap)
            x2 = x1 + card_w

            self._panel(
                frame, x1, card_y,
                x2, card_y + card_h
            )

            cv2.rectangle(
                frame, (x1, card_y),
                (x1 + S(3), card_y + card_h),
                color, -1
            )

            put(
                frame, title,
                (x1 + S(8), card_y + S(19)),
                0.30 * scale, MUTED
            )
            put(
                frame, value,
                (x1 + S(8), card_y + S(47)),
                0.53 * scale, color, S(1)
            )

            if i == 0:
                bar_x = x1 + S(8)
                bar_y = card_y + card_h - S(13)
                bar_w = max(S(12), card_w - S(16))
                cv2.rectangle(
                    frame, (bar_x, bar_y),
                    (bar_x + bar_w, bar_y + S(4)),
                    (65, 58, 78), -1
                )
                cv2.rectangle(
                    frame, (bar_x, bar_y),
                    (bar_x + int(bar_w * self.score / 100),
                     bar_y + S(4)),
                    color, -1
                )

        # Status strip.
        status_y = card_y + card_h + gap
        status_h = S(29)
        self._panel(
            frame, margin, status_y,
            w - margin, status_y + status_h
        )

        elapsed = int(now - self.started)
        mins, secs = divmod(elapsed, 60)

        put(
            frame, f"LOAD STATUS: {level}",
            (margin + S(10), status_y + S(19)),
            0.40 * scale, level_color, S(1)
        )
        put(
            frame, f"SESSION {mins:02d}:{secs:02d}",
            (w - margin - S(150), status_y + S(19)),
            0.36 * scale, WHITE
        )

        # Graph.
        graph_y = status_y + status_h + gap
        available_h = h - graph_y - margin

        if available_h >= S(70):
            self._draw_graph(
                frame, margin, graph_y,
                usable_w, available_h
            )

        # Subtle animated scanning line.
        scan_height = max(1, S(2))
        scan_y = int(
            ((now * S(55)) % max(h, 1))
        )
        if scan_y < h - margin:
            overlay = frame.copy()
            cv2.line(
                overlay, (margin, scan_y),
                (w - margin, scan_y),
                CYAN, scan_height, cv2.LINE_AA
            )
            cv2.addWeighted(
                overlay, 0.18, frame, 0.82, 0, frame
            )

        return frame