"""Anomaly scoring strategies for 3D point cloud inspection.

Three complementary scoring modes
----------------------------------
reconstruction_score  — global Chamfer distance between input and reconstruction.
                        Simple and fast; works with any autoencoder.

pointwise_score       — per-point L2 reconstruction error for anomaly localisation.

MemoryBankScorer      — stores latent embeddings of all *normal* training samples,
                        then scores a test sample by its distance to the k nearest
                        normal neighbours in latent space.
                        Much more powerful than raw reconstruction error alone.
                        Inspired by PatchCore (Roth et al., CVPR 2022).

CombinedScorer        — fuses reconstruction_score + memory_bank_score (normalised)
                        for best AUROC.
"""
import torch
import numpy as np
from src.training.losses import chamfer_distance


# ── Reconstruction-based scoring ──────────────────────────────────────────────

@torch.no_grad()
def reconstruction_score(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: str = "cpu",
) -> np.ndarray:
    """
    Per-sample anomaly score = Chamfer distance(input, reconstruction).

    Args:
        model : trained autoencoder  (forward returns (recon, ...) tuple)
        x     : (B, N, 3)
        device: torch device string
    Returns:
        scores: (B,) float32 numpy array
    """
    model.eval()
    x = x.to(device)
    output = model(x)
    recon  = output[0] if isinstance(output, tuple) else output

    scores = []
    for i in range(x.shape[0]):
        cd = chamfer_distance(recon[i:i+1], x[i:i+1])
        scores.append(cd.item())
    return np.array(scores, dtype=np.float32)


# ── Point-wise localisation score ─────────────────────────────────────────────

@torch.no_grad()
def pointwise_score(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: str = "cpu",
) -> np.ndarray:
    """
    Per-point anomaly score for localisation.

    Returns the squared L2 distance between each reconstructed point
    and its corresponding input point.

    Args:
        model : trained autoencoder
        x     : (B, N, 3)
    Returns:
        scores: (B, N) float32 numpy array
    """
    model.eval()
    x = x.to(device)
    output = model(x)
    recon  = output[0] if isinstance(output, tuple) else output
    diff   = (recon - x).pow(2).sum(-1)    # (B, N)
    return diff.cpu().numpy().astype(np.float32)


# ── Memory Bank scorer ────────────────────────────────────────────────────────

class MemoryBankScorer:
    """
    Stores latent embeddings of normal training samples and scores
    test samples by k-NN distance in that latent space.

    Usage
    -----
    scorer = MemoryBankScorer(model, k=5, device="cuda")
    scorer.fit(train_loader)            # build the bank
    scores = scorer.score(test_loader)  # (N_test,) anomaly scores
    """

    def __init__(
        self,
        model: torch.nn.Module,
        k: int = 5,
        device: str = "cpu",
        subsample: int = 0,
    ):
        """
        Args:
            model    : trained encoder (must expose an .encode() method, or
                       the full forward() whose second output is the latent z)
            k        : number of nearest normal neighbours used for scoring
            device   : torch device
            subsample: if > 0, randomly sub-sample the bank to this size
                       (speeds up search for large datasets)
        """
        self.model     = model.to(device)
        self.k         = k
        self.device    = device
        self.subsample = subsample
        self.bank: torch.Tensor = None   # (M, D)

    # ── Build ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def fit(self, loader) -> None:
        """
        Encode all samples in `loader` and store their embeddings.

        Args:
            loader: DataLoader yielding (points, label) or (points,) batches.
                    Only *normal* samples should be passed (train split).
        """
        self.model.eval()
        embeddings = []

        for batch in loader:
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(self.device)
            z = self._encode(x)              # (B, D)
            embeddings.append(z.cpu())

        bank = torch.cat(embeddings, dim=0)  # (M, D)

        if self.subsample and self.subsample < bank.shape[0]:
            idx  = torch.randperm(bank.shape[0])[:self.subsample]
            bank = bank[idx]

        # L2-normalise for cosine-like distance
        self.bank = torch.nn.functional.normalize(bank, dim=-1)

    # ── Score ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def score(self, loader) -> np.ndarray:
        """
        Compute anomaly scores for all samples in `loader`.

        Returns:
            scores: (N,) float32 — higher = more anomalous
        """
        if self.bank is None:
            raise RuntimeError("Call .fit() before .score()")

        self.model.eval()
        bank   = self.bank.to(self.device)  # (M, D)
        scores = []

        for batch in loader:
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(self.device)
            z = torch.nn.functional.normalize(self._encode(x), dim=-1)  # (B, D)

            # Pairwise cosine distance: 1 - (z @ bank^T)  in [0, 2]
            sim  = z @ bank.T                              # (B, M)
            dist = 1.0 - sim                               # (B, M)

            # Mean of k smallest distances
            knn_d, _ = dist.topk(self.k, dim=-1, largest=False)
            score    = knn_d.mean(dim=-1)                  # (B,)
            scores.append(score.cpu().numpy())

        return np.concatenate(scores, axis=0).astype(np.float32)

    # ── Internal encode helper ────────────────────────────────────────────────

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Route to .encode() if available, else use forward()[1]."""
        if hasattr(self.model, "encode"):
            return self.model.encode(x)
        out = self.model(x)
        return out[1] if isinstance(out, tuple) else out


# ── Combined scorer ───────────────────────────────────────────────────────────

class CombinedScorer:
    """
    Fuses reconstruction score and memory-bank score.

    Both components are min-max normalised to [0, 1] before fusion,
    then combined as:
        combined = alpha * recon_score + (1 - alpha) * bank_score

    Default alpha=0.4 weights the memory bank more, which tends to give
    better AUROC on MVTec 3D-AD.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        k: int = 5,
        alpha: float = 0.4,
        device: str = "cpu",
        subsample: int = 0,
    ):
        self.model  = model
        self.alpha  = alpha
        self.device = device
        self._bank  = MemoryBankScorer(model, k=k, device=device, subsample=subsample)
        self._recon_ref: np.ndarray = None   # reconstruction scores on train set (for normalisation)

    @torch.no_grad()
    def fit(self, train_loader) -> None:
        """Build memory bank and collect reference reconstruction scores."""
        self._bank.fit(train_loader)

        # Collect reconstruction scores on train set for normalisation
        self.model.eval()
        scores = []
        for batch in train_loader:
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch)
            s = reconstruction_score(self.model, x, self.device)
            scores.append(s)
        self._recon_ref = np.concatenate(scores)

    @torch.no_grad()
    def score(self, loader) -> np.ndarray:
        """Return combined anomaly scores for all samples in `loader`."""
        if self._recon_ref is None:
            raise RuntimeError("Call .fit() before .score()")

        # Reconstruction scores
        self.model.eval()
        r_scores = []
        for batch in loader:
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch)
            r_scores.append(reconstruction_score(self.model, x, self.device))
        r_scores = np.concatenate(r_scores)

        # Memory bank scores
        b_scores = self._bank.score(loader)

        # Normalise each component to [0, 1]
        def _norm(arr, ref=None):
            src = ref if ref is not None else arr
            mn, mx = src.min(), src.max()
            return (arr - mn) / (mx - mn + 1e-8)

        r_norm = _norm(r_scores, self._recon_ref)
        b_norm = _norm(b_scores)

        return (self.alpha * r_norm + (1.0 - self.alpha) * b_norm).astype(np.float32)
