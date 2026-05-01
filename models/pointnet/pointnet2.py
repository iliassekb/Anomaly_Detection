"""PointNet++ hierarchical encoder with Set Abstraction layers.

Qi et al., "PointNet++: Deep Hierarchical Feature Learning on Point Sets
in a Metric Space", NeurIPS 2017.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Geometric helpers ─────────────────────────────────────────────────────────

def farthest_point_sample(xyz: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS).
    Args:
        xyz      : (B, N, 3)
        n_samples: number of centroids to select
    Returns:
        indices  : (B, n_samples) long
    """
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    for i in range(n_samples):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B, device=device), farthest].unsqueeze(1)  # (B,1,3)
        dist = ((xyz - centroid) ** 2).sum(-1)                                  # (B, N)
        distance = torch.min(distance, dist)
        farthest = distance.argmax(dim=1)

    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather points by index.
    Args:
        points: (B, N, C)
        idx   : (B, ...) long
    Returns:
        (B, ..., C)
    """
    B = points.shape[0]
    device = points.device
    view_shape = [B] + [1] * (idx.dim() - 1)
    batch_idx = torch.arange(B, device=device).view(view_shape).expand_as(idx)
    return points[batch_idx, idx]


def ball_query(radius: float, n_samples: int,
               xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    Ball-query neighbourhood grouping.
    Args:
        radius   : search radius
        n_samples: max neighbours to keep
        xyz      : (B, N, 3) all points
        new_xyz  : (B, S, 3) centroids
    Returns:
        group_idx: (B, S, n_samples) long
    """
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    device = xyz.device

    # Pairwise squared distances (B, S, N)
    sq_dist = ((new_xyz.unsqueeze(2) - xyz.unsqueeze(1)) ** 2).sum(-1)

    group_idx = torch.arange(N, device=device).view(1, 1, N).expand(B, S, N).clone()
    group_idx[sq_dist > radius ** 2] = N          # mark out-of-radius as N
    group_idx, _ = group_idx.sort(dim=-1)
    group_idx = group_idx[:, :, :n_samples]        # take closest n_samples

    # Replace padding (value == N) with the first valid neighbour
    group_first = group_idx[:, :, 0:1].expand_as(group_idx)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


# ── Set Abstraction layer ─────────────────────────────────────────────────────

class SetAbstraction(nn.Module):
    """
    PointNet++ Set Abstraction (SA) layer.

    Samples `n_points` centroids via FPS, groups neighbours within `radius`,
    and applies a shared PointNet (Conv2d) to each group.
    """

    def __init__(self, n_points: int, radius: float, n_samples: int,
                 in_channel: int, mlp_channels: list):
        super().__init__()
        self.n_points = n_points
        self.radius = radius
        self.n_samples = n_samples

        layers = []
        last_ch = in_channel + 3          # concatenate relative XYZ
        for out_ch in mlp_channels:
            layers += [
                nn.Conv2d(last_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            last_ch = out_ch
        self.mlp = nn.Sequential(*layers)
        self.out_channels = last_ch

    def forward(self, xyz: torch.Tensor, points: torch.Tensor = None):
        """
        Args:
            xyz   : (B, N, 3)  point positions
            points: (B, N, C)  point features (or None)
        Returns:
            new_xyz   : (B, S, 3)
            new_points: (B, S, C_out)
        """
        B = xyz.shape[0]

        # 1. FPS sampling
        fps_idx = farthest_point_sample(xyz, self.n_points)   # (B, S)
        new_xyz = index_points(xyz, fps_idx)                   # (B, S, 3)

        # 2. Ball-query grouping
        group_idx = ball_query(self.radius, self.n_samples, xyz, new_xyz)  # (B,S,K)
        grouped_xyz = index_points(xyz, group_idx)             # (B, S, K, 3)
        grouped_xyz -= new_xyz.unsqueeze(2)                    # relative coords

        if points is not None:
            grouped_pts = index_points(points, group_idx)      # (B, S, K, C)
            grouped_pts = torch.cat([grouped_xyz, grouped_pts], dim=-1)
        else:
            grouped_pts = grouped_xyz                          # (B, S, K, 3)

        # 3. Shared PointNet  (B, C_in, K, S) → (B, C_out, K, S) → max → (B,S,C_out)
        grouped_pts = grouped_pts.permute(0, 3, 2, 1)         # (B, C_in, K, S)
        grouped_pts = self.mlp(grouped_pts)                    # (B, C_out, K, S)
        new_points = grouped_pts.max(dim=2).values             # (B, C_out, S)
        new_points = new_points.permute(0, 2, 1)               # (B, S, C_out)

        return new_xyz, new_points


# ── PointNet++ Encoder ────────────────────────────────────────────────────────

class PointNetPPEncoder(nn.Module):
    """
    Three-level PointNet++ encoder.

    Level 1 : 2048 → 512  points,  radius 0.2,  features 128
    Level 2 : 512  → 128  points,  radius 0.4,  features 256
    Level 3 : 128  → 32   points,  radius 0.8,  features 512

    Global max-pool + FC head → latent vector of size `latent_dim`.
    """

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.sa1 = SetAbstraction(512,  0.2, 32, in_channel=0,   mlp_channels=[64,  64,  128])
        self.sa2 = SetAbstraction(128,  0.4, 64, in_channel=128, mlp_channels=[128, 128, 256])
        self.sa3 = SetAbstraction(32,   0.8, 64, in_channel=256, mlp_channels=[256, 256, 512])

        self.head = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, latent_dim),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """xyz: (B, N, 3)  →  z: (B, latent_dim)"""
        xyz1, f1 = self.sa1(xyz,  None)
        xyz2, f2 = self.sa2(xyz1, f1)
        _,    f3 = self.sa3(xyz2, f2)

        g = f3.max(dim=1).values     # global max-pool  (B, 512)
        return self.head(g)


# ── PointNet++ VAE Encoder ────────────────────────────────────────────────────

class PointNetPPVAEEncoder(nn.Module):
    """Same as PointNetPPEncoder but outputs (mu, log_var) for a VAE."""

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.sa1 = SetAbstraction(512,  0.2, 32, in_channel=0,   mlp_channels=[64,  64,  128])
        self.sa2 = SetAbstraction(128,  0.4, 64, in_channel=128, mlp_channels=[128, 128, 256])
        self.sa3 = SetAbstraction(32,   0.8, 64, in_channel=256, mlp_channels=[256, 256, 512])

        self.shared = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.fc_mu     = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, xyz: torch.Tensor):
        """xyz: (B, N, 3)  →  mu, log_var  each (B, latent_dim)"""
        xyz1, f1 = self.sa1(xyz,  None)
        xyz2, f2 = self.sa2(xyz1, f1)
        _,    f3 = self.sa3(xyz2, f2)

        g = f3.max(dim=1).values
        h = self.shared(g)
        return self.fc_mu(h), self.fc_logvar(h)
