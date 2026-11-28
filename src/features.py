"""Shared MediaPipe hand landmark extraction with model-specific normalization."""
import cv2
import mediapipe as mp
import numpy as np

_mp_hands = mp.solutions.hands


def extract_raw_hand_landmarks(image_bgr):
    """Return raw 63 XYZ landmark values, or None."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with _mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:
        result = hands.process(image_rgb)

    if not result.multi_hand_landmarks:
        return None

    points = np.array(
        [[lm.x, lm.y, lm.z]
         for lm in result.multi_hand_landmarks[0].landmark],
        dtype=np.float32
    )
    return points.flatten()


def normalize_bdsl(values):
    """BdSL47: wrist origin, maximum 3D Euclidean distance."""
    points = np.asarray(values, dtype=np.float32).reshape(21, 3)
    points = points - points[0].copy()
    scale = np.max(np.linalg.norm(points, axis=1))

    if scale < 1e-6:
        raise ValueError("Invalid BdSL hand landmarks")

    return (points / scale).flatten().astype(np.float32)


def normalize_asl(values):
    """ASL repo: wrist origin, wrist-to-middle-MCP scale."""
    points = np.asarray(values, dtype=np.float32).reshape(21, 3)
    points = points - points[0].copy()
    scale = np.linalg.norm(points[9] - points[0])

    if scale < 1e-6:
        raise ValueError("Invalid ASL hand landmarks")

    return (points / scale).flatten().astype(np.float32)


def extract_hand_features(image_bgr):
    """Backward-compatible BdSL feature extraction."""
    raw = extract_raw_hand_landmarks(image_bgr)
    return None if raw is None else normalize_bdsl(raw)
