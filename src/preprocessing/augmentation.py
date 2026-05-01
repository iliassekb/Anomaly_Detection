"""Data augmentation for 3D industrial point clouds (numpy-based).

All transforms receive and return  (N, 3)  float32 numpy arrays so they
compose cleanly with the rest of the preprocessing pipeline.

Available transforms
--------------------
rotate_z          — full 360° rotation around the vertical axis (Z).
                    Industrial parts are typically placed on a flat surface,
                    so Z-rotation is always valid.

rotate_small      — small random tilt on X/Y (±max_deg degrees).
                    Simulates slight sensor mis-alignment.

jitter            — per-point Gaussian noise.  σ=0.005 (0.5% of unit sphere).

scale             — uniform random scaling within [low, high].
                    Simulates part size variations and sensor distance changes.

random_dropout    — randomly zeroes / removes a fraction of points and
                    replaces them with duplicates of surviving points.
                    Simulates scanner occlusion and missing data.

height_noise      — adds structured noise along the Z axis only.
                    Simulates surface roughness variations.

compose           — helper to chain multiple transforms.

augment_train     — recommended default pipeline for training.
"""
import numpy as np


# ── Rotation helpers ──────────────────────────────────────────────────────────

def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)

def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


# ── Individual transforms ─────────────────────────────────────────────────────

def rotate_z(points: np.ndarray) -> np.ndarray:
    """Full 360° random rotation around Z (vertical axis).

    Industrial parts placed on a flat surface can appear at any in-plane
    orientation — this is the most important augmentation.
    """
    angle = np.random.uniform(0.0, 2.0 * np.pi)
    return points @ _rot_z(angle).T


def rotate_small(points: np.ndarray, max_deg: float = 15.0) -> np.ndarray:
    """Small random tilt on X and Y axes (±max_deg degrees).

    Simulates slight sensor mis-alignment or surface inclination.
    Default ±15° is realistic for structured-light / ToF scanners.
    """
    limit = np.deg2rad(max_deg)
    ax = np.random.uniform(-limit, limit)
    ay = np.random.uniform(-limit, limit)
    R  = _rot_y(ay) @ _rot_x(ax)
    return points @ R.T


def jitter(points: np.ndarray, sigma: float = 0.005,
           clip: float = 0.02) -> np.ndarray:
    """Per-point Gaussian noise.

    Args:
        sigma: noise standard deviation (relative to unit sphere).
               0.005 = 0.5% — appropriate for high-precision scans.
        clip : hard clamp to avoid extreme outliers.
    """
    noise = np.random.normal(0.0, sigma, points.shape).astype(np.float32)
    noise = np.clip(noise, -clip, clip)
    return points + noise


def scale(points: np.ndarray, low: float = 0.9,
          high: float = 1.1) -> np.ndarray:
    """Uniform random scaling within [low, high].

    Simulates part size tolerance and varying sensor-to-object distance.
    """
    s = np.random.uniform(low, high)
    return (points * s).astype(np.float32)


def random_dropout(points: np.ndarray, max_ratio: float = 0.1) -> np.ndarray:
    """Randomly drop up to max_ratio of points and fill with duplicates.

    Simulates scanner occlusion, reflective surfaces, and missing data
    patches.  The output always has the same number of points N.

    Args:
        max_ratio: maximum fraction of points to drop (0.10 = 10%).
    """
    N = len(points)
    drop_ratio = np.random.uniform(0.0, max_ratio)
    n_drop = int(N * drop_ratio)
    if n_drop == 0:
        return points

    keep_idx = np.random.choice(N, N - n_drop, replace=False)
    kept     = points[keep_idx]

    # Fill dropped positions by duplicating random kept points
    fill_idx = np.random.choice(len(kept), n_drop, replace=True)
    result   = np.concatenate([kept, kept[fill_idx]], axis=0)

    # Shuffle so dropped positions are not always at the end
    perm = np.random.permutation(N)
    return result[perm]


def height_noise(points: np.ndarray, sigma: float = 0.003) -> np.ndarray:
    """Structured noise along the Z axis only.

    Simulates surface roughness, micro-deformations, and sensor depth
    noise which are predominantly vertical (height) in industrial scans.
    """
    noise      = np.zeros_like(points)
    noise[:, 2] = np.random.normal(0.0, sigma, len(points)).astype(np.float32)
    return points + noise


# ── Compose ───────────────────────────────────────────────────────────────────

def compose(transforms: list):
    """Chain multiple augmentation functions into a single callable.

    Usage:
        aug = compose([rotate_z, jitter, scale])
        augmented = aug(points)
    """
    def _apply(points: np.ndarray) -> np.ndarray:
        for t in transforms:
            points = t(points)
        return points
    return _apply


# ── Default training pipeline ─────────────────────────────────────────────────

def augment_train(points: np.ndarray) -> np.ndarray:
    """Recommended augmentation pipeline for industrial anomaly detection.

    Applied transforms (in order):
      1. rotate_z      — in-plane 360° rotation (always applied)
      2. rotate_small  — ±15° tilt on X/Y     (always applied)
      3. jitter        — per-point noise σ=0.005 (always applied)
      4. scale         — [0.9, 1.1] uniform    (applied with p=0.7)
      5. random_dropout— up to 5% points lost  (applied with p=0.5)
      6. height_noise  — Z-axis surface noise  (applied with p=0.5)

    Probabilities are tuned so augmentation is strong enough to improve
    generalisation without distorting the normal geometry too much.
    """
    # Always applied
    points = rotate_z(points)
    points = rotate_small(points, max_deg=15.0)
    points = jitter(points, sigma=0.005)

    # Stochastic transforms
    if np.random.rand() < 0.7:
        points = scale(points, low=0.9, high=1.1)

    if np.random.rand() < 0.5:
        points = random_dropout(points, max_ratio=0.05)

    if np.random.rand() < 0.5:
        points = height_noise(points, sigma=0.003)

    return points.astype(np.float32)
