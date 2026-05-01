"""Point Cloud Autoencoder for unsupervised anomaly detection.

Two backends are provided:

* SimplePointCloudAutoEncoder  — original MLP encoder + MLP decoder
  (kept for ablation / fast baselines)

* PointCloudAutoEncoder  (default)  — PointNet++ encoder + FoldingNet decoder
  Significantly better reconstruction of local geometry, which translates
  to higher AUROC on anomaly-detection benchmarks.
"""
import torch
import torch.nn as nn

from models.pointnet.pointnet2 import PointNetPPEncoder
from models.autoencoders.folding_decoder import FoldingNetDecoder


# ── Simple baseline (kept for ablation) ───────────────────────────────────────

class _SimpleEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),   nn.ReLU(inplace=True),
            nn.Linear(64, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 256),nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.mlp(x).max(dim=1).values
        return self.fc(feat)


class _SimpleDecoder(nn.Module):
    def __init__(self, latent_dim: int = 128, n_points: int = 2048):
        super().__init__()
        self.n_points = n_points
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 512),        nn.ReLU(inplace=True),
            nn.Linear(512, n_points * 3),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z).view(-1, self.n_points, 3)


class SimplePointCloudAutoEncoder(nn.Module):
    """Lightweight baseline — fast to train, weaker reconstruction."""

    def __init__(self, latent_dim: int = 128, n_points: int = 2048):
        super().__init__()
        self.encoder = _SimpleEncoder(latent_dim)
        self.decoder = _SimpleDecoder(latent_dim, n_points)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return self.decoder(z), z


# ── Main model: PointNet++ encoder + FoldingNet decoder ───────────────────────

class PointCloudAutoEncoder(nn.Module):
    """
    State-of-the-art autoencoder for 3-D anomaly detection.

    Encoder : PointNet++ with three Set-Abstraction levels.
              Captures hierarchical local geometry (edges, surfaces, global shape).

    Decoder : FoldingNet — two-stage MLP folding of a 2-D grid into 3-D space.
              Produces smooth, well-distributed reconstructions.

    Anomaly score at inference: Chamfer distance between input and reconstruction.
    """

    def __init__(self, latent_dim: int = 256, n_points: int = 2048):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_points   = n_points
        self.encoder    = PointNetPPEncoder(latent_dim)
        self.decoder    = FoldingNetDecoder(latent_dim, n_points)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, 3)  →  z: (B, latent_dim)"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim)  →  (B, N, 3)"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            recon : (B, N, 3)  reconstructed point cloud
            z     : (B, D)     latent code
        """
        z     = self.encode(x)
        recon = self.decode(z)
        return recon, z
