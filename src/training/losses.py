"""Loss functions for 3D anomaly detection.

Available losses
----------------
chamfer_distance    — bidirectional Chamfer Distance (primary reconstruction loss)
density_loss        — penalises uneven point distribution in reconstructions
combined_ae_loss    — Chamfer + density regularisation
vae_loss            — combined_ae_loss + KL divergence (beta-VAE)
"""
import torch


# ── Chamfer Distance ──────────────────────────────────────────────────────────

def chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Symmetric Chamfer Distance.

    For each point in `pred` find its nearest neighbour in `target` and
    vice versa.  Returns the mean of both directed distances.

    Args:
        pred  : (B, N, 3)
        target: (B, M, 3)
    Returns:
        scalar loss
    """
    # (B, N, M, 3)
    diff = pred.unsqueeze(2) - target.unsqueeze(1)
    dist = diff.pow(2).sum(-1)                          # (B, N, M)
    loss = dist.min(dim=2).values.mean() + dist.min(dim=1).values.mean()
    return loss


# ── Density regularisation ────────────────────────────────────────────────────

def density_loss(pred: torch.Tensor, n_neighbours: int = 10) -> torch.Tensor:
    """
    Encourage uniform point distribution by penalising high variance
    in nearest-neighbour distances within the reconstruction.

    A perfectly uniform point cloud has zero variance; clustered outputs
    have high variance.

    Args:
        pred        : (B, N, 3)
        n_neighbours: number of local neighbours to consider
    Returns:
        scalar regularisation term
    """
    N = pred.shape[1]
    k = min(n_neighbours, N - 1)

    # Pairwise distances (B, N, N)
    diff  = pred.unsqueeze(2) - pred.unsqueeze(1)
    dist2 = diff.pow(2).sum(-1)                         # (B, N, N)

    # k smallest distances (exclude self = 0)
    knn_d, _ = dist2.topk(k + 1, dim=-1, largest=False, sorted=True)
    knn_d = knn_d[:, :, 1:]                             # drop self  (B, N, k)

    # Variance of knn distances per point, then mean
    return knn_d.var(dim=-1).mean()


# ── Combined AE loss ──────────────────────────────────────────────────────────

def combined_ae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    density_weight: float = 0.1,
) -> torch.Tensor:
    """
    Chamfer Distance  +  density regularisation.

    Args:
        pred          : (B, N, 3)  reconstruction
        target        : (B, N, 3)  input
        density_weight: weight for the density term (lambda_d)
    Returns:
        total loss (scalar)
    """
    cd = chamfer_distance(pred, target)
    dd = density_loss(pred)
    return cd + density_weight * dd


# ── VAE loss ──────────────────────────────────────────────────────────────────

def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
    density_weight: float = 0.05,
):
    """
    Beta-VAE loss: reconstruction + KL + (optional) density regularisation.

    Args:
        recon         : (B, N, 3)  decoder output
        x             : (B, N, 3)  input point cloud
        mu            : (B, D)     posterior mean
        log_var       : (B, D)     posterior log-variance
        beta          : KL weight  (1.0 = standard VAE, <1 focuses on reconstruction)
        density_weight: weight for density term
    Returns:
        total, recon_loss, kl  (all scalars)
    """
    recon_loss = combined_ae_loss(recon, x, density_weight)
    kl         = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    total      = recon_loss + beta * kl
    return total, recon_loss, kl
