import time
from collections import deque
import numpy as np

class CognitiveLoadEstimator:
    def __init__(self, window_seconds=15):
        self.window_seconds = window_seconds

        # Each deque entry: (timestamp, value)
        self.blink_events = deque()      # timestamps of detected blinks
        self.gaze_samples = deque()      # (timestamp, gaze_offset)
        self.headmove_samples = deque()  # (timestamp, movement_magnitude)
        self.ear_samples = deque()       # (timestamp, ear) - for drowsiness

        self._last_head_pose = None

    # ------------------------------------------------------------------
    # Feed new data in every frame
    # ------------------------------------------------------------------
    def add_blink(self, timestamp=None):
        t = timestamp or time.time()
        self.blink_events.append(t)
        self._trim(self.blink_events)

    def add_gaze_sample(self, gaze_value, timestamp=None):
        t = timestamp or time.time()
        self.gaze_samples.append((t, gaze_value))
        self._trim(self.gaze_samples)

    def add_head_pose(self, pitch, yaw, roll, timestamp=None):
        t = timestamp or time.time()
        if self._last_head_pose is not None:
            p0, y0, r0 = self._last_head_pose
            movement = abs(pitch - p0) + abs(yaw - y0) + abs(roll - r0)
        else:
            movement = 0.0
        self._last_head_pose = (pitch, yaw, roll)
        self.headmove_samples.append((t, movement))
        self._trim(self.headmove_samples)

    def add_ear_sample(self, ear_value, timestamp=None):
        t = timestamp or time.time()
        self.ear_samples.append((t, ear_value))
        self._trim(self.ear_samples)

    # ------------------------------------------------------------------
    def _trim(self, dq):
        """Drop samples older than window_seconds."""
        cutoff = time.time() - self.window_seconds
        while dq and (dq[0][0] if isinstance(dq[0], tuple) else dq[0]) < cutoff:
            dq.popleft()

    # ------------------------------------------------------------------
    def blink_rate_per_min(self):
        self._trim(self.blink_events)
        n = len(self.blink_events)
        minutes = self.window_seconds / 60.0
        return n / minutes if minutes > 0 else 0.0

    def gaze_variance(self):
        self._trim(self.gaze_samples)
        if len(self.gaze_samples) < 2:
            return 0.0
        values = np.array([v for _, v in self.gaze_samples])
        return float(np.std(values))

    def avg_head_movement(self):
        self._trim(self.headmove_samples)
        if not self.headmove_samples:
            return 0.0
        values = np.array([v for _, v in self.headmove_samples])
        return float(np.mean(values))

    def drowsiness_flag(self, ear_threshold=0.20, min_seconds_closed=1.5):
        """
        Detects sustained eye closure (drowsiness), not just a quick blink.
        A quick blink lasts ~0.1-0.4s; drowsy eye closure lasts much longer.
        """
        self._trim(self.ear_samples)
        if not self.ear_samples:
            return False

        closed_streak_start = None
        for t, ear in reversed(self.ear_samples):
            if ear < ear_threshold:
                closed_streak_start = t
            else:
                break

        if closed_streak_start is None:
            return False

        latest_t = self.ear_samples[-1][0]
        return (latest_t - closed_streak_start) >= min_seconds_closed

    # ------------------------------------------------------------------
    def compute_load(self):
        """
        Returns a dict:
            {
              'score': 0-100,
              'label': 'Low' | 'Medium' | 'High',
              'blink_rate': float,
              'gaze_variance': float,
              'head_movement': float,
              'drowsy': bool
            }

        Scoring (heuristic, tune freely):
          - Normal blink rate at rest: ~15-20/min.
              Too low (<8) OR too high (>25) both push load up
              (too low => over-focus/strain, too high => fatigue/stress)
          - High gaze variance => attention drifting => load up
          - High head movement => restlessness/distraction => load up
          - Drowsiness => load forced to a high "fatigue" score
        """
        blink_rate = self.blink_rate_per_min()
        gaze_var = self.gaze_variance()
        head_move = self.avg_head_movement()
        drowsy = self.drowsiness_flag()

        # --- sub-scores, each normalized to roughly 0-100 ---
        # blink component: distance from the healthy 15-20/min band
        if blink_rate < 8:
            blink_score = min(100, (8 - blink_rate) * 12)
        elif blink_rate > 25:
            blink_score = min(100, (blink_rate - 25) * 8)
        else:
            blink_score = 0

        gaze_score = min(100, gaze_var * 400)       # gaze_var is typically 0-0.25
        head_score = min(100, head_move * 15)        # head_move in degrees/frame-delta

        # weighted combination
        score = (0.4 * blink_score) + (0.35 * gaze_score) + (0.25 * head_score)
        score = float(np.clip(score, 0, 100))

        if drowsy:
            score = max(score, 85)  # drowsiness overrides everything else

        if score < 35:
            label = "Low"
        elif score < 65:
            label = "Medium"
        else:
            label = "High"

        return {
            "score": round(score, 1),
            "label": label,
            "blink_rate": round(blink_rate, 1),
            "gaze_variance": round(gaze_var, 4),
            "head_movement": round(head_move, 2),
            "drowsy": drowsy,
        }