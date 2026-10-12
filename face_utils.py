import cv2
import numpy as np

# FaceMesh landmark indices

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

POSE_LANDMARKS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 33,
    "right_eye_corner": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

MODEL_3D_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)


def landmark_to_xy(landmark, w, h):
    return np.array([landmark.x * w, landmark.y * h])


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [landmark_to_xy(landmarks[i], w, h) for i in eye_indices]

    p1, p2, p3, p4, p5, p6 = pts

    vertical1 = np.linalg.norm(p2 - p6)
    vertical2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal == 0:
        return 0.0

    return (vertical1 + vertical2) / (2.0 * horizontal)


def average_ear(landmarks, w, h):
    left = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
    right = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
    return (left + right) / 2.0


def gaze_offset(landmarks, w, h):

    def eye_offset(iris_idx, eye_idx):
        iris = np.array([landmark_to_xy(landmarks[i], w, h) for i in iris_idx])
        iris_center = iris.mean(axis=0)

        eye = np.array([landmark_to_xy(landmarks[i], w, h) for i in eye_idx])
        eye_center = eye.mean(axis=0)

        eye_width = np.linalg.norm(eye[0] - eye[3]) + 1e-6

        return np.linalg.norm(iris_center - eye_center) / eye_width

    left = eye_offset(LEFT_IRIS, LEFT_EYE)
    right = eye_offset(RIGHT_IRIS, RIGHT_EYE)

    return (left + right) / 2.0


def head_pose(landmarks, w, h):

    image_points = np.array([
        landmark_to_xy(landmarks[POSE_LANDMARKS["nose_tip"]], w, h),
        landmark_to_xy(landmarks[POSE_LANDMARKS["chin"]], w, h),
        landmark_to_xy(landmarks[POSE_LANDMARKS["left_eye_corner"]], w, h),
        landmark_to_xy(landmarks[POSE_LANDMARKS["right_eye_corner"]], w, h),
        landmark_to_xy(landmarks[POSE_LANDMARKS["left_mouth"]], w, h),
        landmark_to_xy(landmarks[POSE_LANDMARKS["right_mouth"]], w, h),
    ], dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_3D_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    projection_matrix = np.hstack((rotation_matrix, translation_vector))
    euler = cv2.decomposeProjectionMatrix(projection_matrix)[6]

    pitch, yaw, roll = [float(x) for x in euler.flatten()]

    return pitch, yaw, roll