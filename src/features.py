"""Shared MediaPipe landmark extraction and normalization."""
import cv2
import mediapipe as mp
import numpy as np

_mp_hands = mp.solutions.hands

def extract_hand_features(image_bgr):
    """Return 63 normalized XYZ values, or None if no hand is detected."""
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
        [[lm.x, lm.y, lm.z] for lm in result.multi_hand_landmarks[0].landmark],
        dtype=np.float32
    )
    # Translate wrist to origin, then scale by the largest XY spread.
    points -= points[0]
    scale = np.max(np.linalg.norm(points[:, :2], axis=1))
    if scale > 1e-6:
        points /= scale
    return points.flatten()
