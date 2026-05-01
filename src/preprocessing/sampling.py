"""Point cloud sampling strategies (numpy-based)."""
import numpy as np


def farthest_point_sample_np(points: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Farthest Point Sampling. Returns indices of selected points.
    points: (N, 3)
    """
    N = len(points)
    selected = np.zeros(n_samples, dtype=int)
    distances = np.full(N, np.inf)
    current = np.random.randint(0, N)
    for i in range(n_samples):
        selected[i] = current
        dist = np.linalg.norm(points - points[current], axis=1)
        distances = np.minimum(distances, dist)
        current = np.argmax(distances)
    return selected


def random_sample_np(points: np.ndarray, n_samples: int) -> np.ndarray:
    replace = len(points) < n_samples
    return np.random.choice(len(points), n_samples, replace=replace)
