"""MVTec 3D-AD dataset loader.

Data format:
  data/raw/<category>/<split>/<defect_type>/xyz/*.tiff   ← (H, W, 3) float32 XYZ maps
  data/raw/<category>/<split>/<defect_type>/rgb/*.png    ← colour images (optional)
  data/raw/<category>/test/<defect_type>/gt/*.png        ← binary anomaly masks (test only)
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile
from PIL import Image

from src.preprocessing.normalize import center_and_scale
from src.preprocessing.sampling import farthest_point_sample_np
from src.preprocessing.augmentation import augment_train


class MVTec3DDataset(Dataset):
    """
    Args:
        root       : path to data/raw
        category   : e.g. "bagel"
        split      : "train" | "validation" | "test"
        n_points   : number of points to sample from each scan
        augment    : apply full augmentation pipeline (train only)
        return_mask: also return per-point GT anomaly mask (test only)
    """

    def __init__(
        self,
        root: str,
        category: str,
        split: str = "train",
        n_points: int = 2048,
        augment: bool = False,
        return_mask: bool = False,
    ):
        self.root        = Path(root)
        self.category    = category
        self.split       = split
        self.n_points    = n_points
        self.augment     = augment
        self.return_mask = return_mask

        self.samples: list[dict] = []
        self._load_samples()

    # ── Index building ────────────────────────────────────────────────────────

    def _load_samples(self):
        split_dir = self.root / self.category / self.split
        for defect_dir in sorted(split_dir.iterdir()):
            if not defect_dir.is_dir():
                continue
            label   = 0 if defect_dir.name == "good" else 1
            xyz_dir = defect_dir / "xyz"
            gt_dir  = defect_dir / "gt"
            if not xyz_dir.exists():
                continue
            for xyz_path in sorted(xyz_dir.glob("*.tiff")):
                stem    = xyz_path.stem
                gt_path = gt_dir / f"{stem}.png" if gt_dir.exists() else None
                self.samples.append({
                    "xyz":    xyz_path,
                    "gt":     gt_path,
                    "label":  label,
                    "defect": defect_dir.name,
                })

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        # 1. Load XYZ map → raw point cloud
        xyz_map = tifffile.imread(str(sample["xyz"]))        # (H, W, 3) float32
        points, valid_mask_flat = xyz_map_to_points(xyz_map)

        # 2. Load GT mask (test split only)
        gt_point_mask = None
        if self.return_mask and sample["gt"] is not None and sample["gt"].exists():
            gt_img        = np.array(Image.open(sample["gt"]).convert("L")) > 0
            gt_point_mask = gt_img.reshape(-1)[valid_mask_flat]

        # 3. Center & scale to unit sphere
        points = center_and_scale(points)

        # 4. Pre-filter to ≤ 20 k points before FPS (speed)
        N = len(points)
        if N > 20_000:
            pre_idx = np.random.choice(N, 20_000, replace=False)
            points  = points[pre_idx]
            if gt_point_mask is not None:
                gt_point_mask = gt_point_mask[pre_idx]

        # 5. FPS to fixed n_points
        if len(points) >= self.n_points:
            idx_s = farthest_point_sample_np(points, self.n_points)
        else:
            idx_s = np.random.choice(len(points), self.n_points, replace=True)
        points = points[idx_s]
        if gt_point_mask is not None:
            gt_point_mask = gt_point_mask[idx_s]

        # 6. Augmentation (training only, applied AFTER sampling so every
        #    call produces a differently augmented view of the same scan)
        if self.augment:
            points = augment_train(points)

        tensor = torch.tensor(points, dtype=torch.float32)
        label  = sample["label"]

        if self.return_mask and gt_point_mask is not None:
            return tensor, label, torch.tensor(gt_point_mask, dtype=torch.float32)
        return tensor, label


# ── File-level helpers ────────────────────────────────────────────────────────

def xyz_map_to_points(xyz_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert (H, W, 3) XYZ map → (N, 3) point cloud, dropping NaN/zero pixels."""
    flat  = xyz_map.reshape(-1, 3)
    valid = ~(np.isnan(flat).any(axis=1) | (flat == 0).all(axis=1))
    return flat[valid].astype(np.float32), valid
