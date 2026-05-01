"""Normalize, center, and scale point clouds."""
import numpy as np


def center_and_scale(points: np.ndarray) -> np.ndarray:
    """Center to origin and scale to unit sphere. points: (N, 3)"""
    points = points - points.mean(axis=0)
    max_dist = np.linalg.norm(points, axis=1).max()
    if max_dist > 0:
        points /= max_dist
    return points
