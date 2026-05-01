"""FoldingNet decoder: learns to fold a 2-D grid into a 3-D point cloud.

Yang et al., "FoldingNet: Point Cloud Auto-encoder via Deep Grid Deformation",
CVPR 2018.

Two successive MLP-based folding stages:
  Stage 1 — (latent + 2D grid)  → coarse 3D shape
  Stage 2 — (latent + stage-1)  → refined 3D shape
"""
import math
import torch
import torch.nn as nn


def _build_grid(n_points: int) -> torch.Tensor:
    """
    Build a 2-D meshgrid of `n_points` positions in [-1, 1]^2.

    If n_points is not a perfect square, we generate the nearest larger
    square grid and uniformly sub-sample down to n_points.

    Returns:
        grid: (n_points, 2)  float32
    """
    side = math.ceil(math.sqrt(n_points))
    lin = torch.linspace(-1.0, 1.0, side)
    gx, gy = torch.meshgrid(lin, lin, indexing="ij")          # (side, side)
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)  # (side^2, 2)

    if grid.shape[0] > n_points:
        # deterministic stride-based sub-sampling
        step = grid.shape[0] / n_points
        idx = torch.arange(n_points).float().mul(step).long()
        grid = grid[idx]

    return grid   # (n_points, 2)


def _fold_mlp(in_ch: int) -> nn.Sequential:
    """Shared MLP used in each folding stage."""
    return nn.Sequential(
        nn.Linear(in_ch, 512, bias=False),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 512, bias=False),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 3),
    )


class FoldingNetDecoder(nn.Module):
    """
    Args:
        latent_dim: dimension of the input latent vector z
        n_points  : number of output points
    """

    def __init__(self, latent_dim: int = 256, n_points: int = 2048):
        super().__init__()
        self.n_points = n_points
        self.latent_dim = latent_dim

        grid = _build_grid(n_points)                 # (N, 2)
        self.register_buffer("grid", grid)            # stays on the right device

        # Fold-1: latent + 2D  → 3D
        self.fold1 = _fold_mlp(latent_dim + 2)
        # Fold-2: latent + 3D  → 3D
        self.fold2 = _fold_mlp(latent_dim + 3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim)
        Returns:
            out: (B, N, 3)
        """
        B, N = z.shape[0], self.n_points

        # Broadcast latent to every point
        z_exp = z.unsqueeze(1).expand(B, N, -1)              # (B, N, D)
        grid  = self.grid.unsqueeze(0).expand(B, -1, -1)     # (B, N, 2)

        # ── Stage 1 ──────────────────────────────────────────────────────────
        inp1  = torch.cat([z_exp, grid], dim=-1)              # (B, N, D+2)
        # Flatten for BN, then fold
        flat1 = inp1.reshape(B * N, -1)
        s1    = self.fold1(flat1).reshape(B, N, 3)            # (B, N, 3)

        # ── Stage 2 ──────────────────────────────────────────────────────────
        inp2  = torch.cat([z_exp, s1], dim=-1)                # (B, N, D+3)
        flat2 = inp2.reshape(B * N, -1)
        s2    = self.fold2(flat2).reshape(B, N, 3)            # (B, N, 3)

        return s2
