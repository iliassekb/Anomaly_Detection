"""Graph-based Autoencoder for 3D point cloud anomaly detection.

Encoder : DGCNN-style Edge Convolutions (dynamic k-NN graph)
          Wang et al., "Dynamic Graph CNN for Learning on Point Clouds",
          ACM TOG 2019.

Decoder : FoldingNet (same as the PointNet++ AE for fair comparison)

Edge convolution captures fine-grained local geometry by building a new
k-NN graph at every layer, making it more expressive than a single
global-pooling PointNet.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.autoencoders.folding_decoder import FoldingNetDecoder


# ── k-NN graph construction ───────────────────────────────────────────────────

def knn_graph(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Build a k-nearest-neighbour graph from point features.

    Args:
        x: (B, N, C)  point feature matrix
        k: number of neighbours
    Returns:
        idx: (B, N, k) long  — neighbour indices for each point
    """
    # Pairwise squared L2 distance
    inner = -2.0 * torch.bmm(x, x.transpose(2, 1))          # (B, N, N)
    sq    = (x ** 2).sum(dim=-1, keepdim=True)               # (B, N, 1)
    dist  = sq + inner + sq.transpose(2, 1)                  # (B, N, N)

    # k+1 to exclude self (distance = 0)
    _, idx = dist.topk(k + 1, dim=-1, largest=False, sorted=True)
    return idx[:, :, 1:]   # (B, N, k)  — drop self


# ── Edge Convolution ──────────────────────────────────────────────────────────

class EdgeConv(nn.Module):
    """
    Edge Convolution layer.

    For each point i with neighbours j1..jk, computes:
        h_i = AGG_j  MLP( [x_i, x_j - x_i] )

    The relative difference (x_j - x_i) encodes local geometry.
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 20):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch * 2, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C)
        Returns:
            (B, N, out_ch)
        """
        B, N, C = x.shape
        k = min(self.k, N - 1)

        idx = knn_graph(x, k)                                     # (B, N, k)

        # Gather neighbours
        idx_flat = idx.reshape(B, -1)                             # (B, N*k)
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(B, N * k)
        neighbours = x[batch_idx, idx_flat].reshape(B, N, k, C)  # (B, N, k, C)

        # Build edge features: [xi, xj - xi]
        xi = x.unsqueeze(2).expand_as(neighbours)                 # (B, N, k, C)
        edge_feat = torch.cat([xi, neighbours - xi], dim=-1)      # (B, N, k, 2C)

        # Conv2d expects (B, C, H, W) → treat N as W, k as H
        edge_feat = edge_feat.permute(0, 3, 2, 1)                 # (B, 2C, k, N)
        out = self.mlp(edge_feat)                                  # (B, out_ch, k, N)
        out = out.max(dim=2).values                                # (B, out_ch, N)
        out = out.permute(0, 2, 1)                                 # (B, N, out_ch)
        return out


# ── DGCNN Encoder ─────────────────────────────────────────────────────────────

class DGCNNEncoder(nn.Module):
    """
    Four EdgeConv layers with residual-like feature concatenation,
    followed by global max-pool and FC bottleneck.
    """

    def __init__(self, latent_dim: int = 256, k: int = 20):
        super().__init__()
        self.ec1 = EdgeConv(3,   64,  k)
        self.ec2 = EdgeConv(64,  128, k)
        self.ec3 = EdgeConv(128, 256, k)
        self.ec4 = EdgeConv(256, 256, k)

        # Aggregate multi-scale features: 64+128+256+256 = 704
        self.head = nn.Sequential(
            nn.Linear(704, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, latent_dim),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """xyz: (B, N, 3)  →  z: (B, latent_dim)"""
        f1 = self.ec1(xyz)                     # (B, N, 64)
        f2 = self.ec2(f1)                      # (B, N, 128)
        f3 = self.ec3(f2)                      # (B, N, 256)
        f4 = self.ec4(f3)                      # (B, N, 256)

        # Multi-scale global max-pool
        g = torch.cat([
            f1.max(dim=1).values,              # (B, 64)
            f2.max(dim=1).values,              # (B, 128)
            f3.max(dim=1).values,              # (B, 256)
            f4.max(dim=1).values,              # (B, 256)
        ], dim=-1)                             # (B, 704)

        return self.head(g)                    # (B, latent_dim)


# ── Graph Autoencoder ─────────────────────────────────────────────────────────

class GraphAutoEncoder(nn.Module):
    """
    Full autoencoder:
        Encoder — DGCNN (4 × EdgeConv + global pool)
        Decoder — FoldingNet (2-stage 2D→3D folding)

    Anomaly score at inference: Chamfer distance between input and reconstruction.
    """

    def __init__(self, latent_dim: int = 256, n_points: int = 2048, k: int = 20):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_points   = n_points
        self.encoder    = DGCNNEncoder(latent_dim, k)
        self.decoder    = FoldingNetDecoder(latent_dim, n_points)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            recon: (B, N, 3)
            z    : (B, latent_dim)
        """
        z     = self.encode(x)
        recon = self.decode(z)
        return recon, z
