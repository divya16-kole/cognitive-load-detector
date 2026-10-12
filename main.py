import time
import cv2
import mediapipe as mp

from face_utils import average_ear, gaze_offset, head_pose
from cognitive_load import CognitiveLoadEstimator
from ui_overlay import Dashboard, put, RED

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EAR_BLINK_THRESHOLD = 0.21      # below this = eye considered closed
BLINK_CONSEC_FRAMES = 2         # min consecutive closed-frames to count as a blink
WINDOW_SECONDS = 15             # rolling window for the load score
GRAPH_SECONDS = 60              # how much history the live graph shows
CAM_INDEX = 0                   # change if you have multiple webcams


def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,   # needed for iris landmarks (gaze)
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check CAM_INDEX or camera permissions.")
        return

    estimator = CognitiveLoadEstimator(window_seconds=WINDOW_SECONDS)
    dash = Dashboard(window_seconds=GRAPH_SECONDS)

    closed_frame_counter = 0
    total_blinks = 0
    session_start = time.time()
    prev_time = time.time()

    print("Cognitive Load Detector running. Press 'q' to quit, 'r' to reset stats.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)  # mirror for natural webcam feel
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        face_found = bool(results.multi_face_landmarks)
        drowsy = False

        if face_found:
            landmarks = results.multi_face_landmarks[0].landmark

            # ---- Blink detection via EAR ----
            ear = average_ear(landmarks, w, h)
            estimator.add_ear_sample(ear)

            if ear < EAR_BLINK_THRESHOLD:
                closed_frame_counter += 1
            else:
                if closed_frame_counter >= BLINK_CONSEC_FRAMES:
                    total_blinks += 1
                    estimator.add_blink()
                closed_frame_counter = 0

            # ---- Gaze drift ----
            gaze = gaze_offset(landmarks, w, h)
            estimator.add_gaze_sample(gaze)

            # ---- Head pose ----
            pose = head_pose(landmarks, w, h)
            if pose is not None:
                pitch, yaw, roll = pose
                estimator.add_head_pose(pitch, yaw, roll)

            # ---- Compute the live score ----
            result = estimator.compute_load()
            drowsy = result["drowsy"]

            # ---- Feed the animated dashboard ----
            dash.update(
                float(result["score"]),
                float(result["blink_rate"]),
                total_blinks,
                float(result["gaze_variance"]),
                float(result["head_movement"]),
            )

        # Dashboard is always drawn (keeps the graph alive even if face is lost)
        frame = dash.draw(frame)

        if not face_found:
            put(frame, "No face detected", (int(w * 0.30), int(h * 0.50)), 0.9 * w / 640, RED, 2)

        if drowsy and int(now * 2) % 2 == 0:
            put(frame, "DROWSINESS DETECTED", (int(w * 0.25), int(h * 0.42)), 0.9 * w / 640, RED, 2)

        put(frame, f"FPS: {fps:.0f}", (w - int(95 * w / 640), int(100 * w / 640) + int(62 * w / 640) + 6),
            0.4 * w / 640, (200, 200, 200), 1)

        cv2.imshow("Cognitive Load Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            estimator = CognitiveLoadEstimator(window_seconds=WINDOW_SECONDS)
            dash = Dashboard(window_seconds=GRAPH_SECONDS)
            total_blinks = 0
            closed_frame_counter = 0
            session_start = time.time()
            print("Session stats reset.")

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()


if __name__ == "__main__":
    main()