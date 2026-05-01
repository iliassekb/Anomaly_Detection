"""Variational Autoencoder for 3D point clouds.

Encoder : PointNet++ (separate mu / log_var heads)
Decoder : FoldingNet (two-stage grid folding)

The VAE is trained with a combined loss:
    L = Chamfer(recon, x) + beta * KL(q(z|x) || p(z))
"""
import torch
import torch.nn as nn

from models.pointnet.pointnet2 import PointNetPPVAEEncoder
from models.autoencoders.folding_decoder import FoldingNetDecoder


class PointCloudVAE(nn.Module):
    """
    Args:
        latent_dim: size of the latent Gaussian
        n_points  : number of output points from the decoder
        beta      : KL weight (beta-VAE formulation; 1.0 = standard VAE)
    """

    def __init__(self, latent_dim: int = 256, n_points: int = 2048, beta: float = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta       = beta
        self.encoder    = PointNetPPVAEEncoder(latent_dim)
        self.decoder    = FoldingNetDecoder(latent_dim, n_points)

    # ── VAE core ─────────────────────────────────────────────────────────────

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample z ~ N(mu, sigma^2) via the reparameterisation trick."""
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, N, 3)
        Returns:
            recon  : (B, N, 3)
            mu     : (B, D)
            log_var: (B, D)
        """
        mu, log_var = self.encoder(x)
        z     = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var

    # ── Convenience: deterministic encode (inference / memory bank) ──────────

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the posterior mean (deterministic embedding for anomaly scoring)."""
        mu, _ = self.encoder(x)
        return mu
