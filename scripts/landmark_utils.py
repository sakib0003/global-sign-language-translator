
import numpy as np


def normalize_landmarks(values):
    """
    Normalize 21 hand landmarks.
    Input: 63 landmark values (x, y, z).
    Output: 63 normalized float32 values.
    """

    points = np.asarray(
        values,
        dtype=np.float32
    ).reshape(21, 3)

    # Move wrist landmark to origin
    points = points - points[0].copy()

    # Calculate distance from wrist
    distances = np.linalg.norm(
        points,
        axis=1
    )

    # Find maximum distance
    scale = np.max(distances)

    if scale < 1e-6:
        raise ValueError(
            "Invalid hand landmarks"
        )

    # Normalize landmarks
    points = points / scale

    return points.flatten().astype(np.float32)
