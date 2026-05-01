"""Entry point: train anomaly detection model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import yaml
import torch
from torch.utils.data import DataLoader, random_split

from src.utils.seed import set_seed
from src.utils.dataset import MVTec3DDataset
from src.training.trainer import Trainer
from src.training.losses import chamfer_distance, combined_ae_loss, vae_loss
from models.autoencoders.pointcloud_ae import PointCloudAutoEncoder, SimplePointCloudAutoEncoder
from models.autoencoders.vae import PointCloudVAE
from models.gnn.graph_ae import GraphAutoEncoder


def build_model(cfg):
    mtype      = cfg["model"]["type"]
    latent_dim = cfg["model"]["latent_dim"]
    n_points   = cfg["dataset"]["n_points"]

    if mtype == "ae":
        return PointCloudAutoEncoder(latent_dim, n_points)
    if mtype == "vae":
        return PointCloudVAE(latent_dim, n_points, beta=cfg["training"].get("beta", 1.0))
    if mtype == "graph_ae":
        return GraphAutoEncoder(latent_dim, n_points)
    if mtype == "simple_ae":
        return SimplePointCloudAutoEncoder(latent_dim, n_points)
    raise ValueError(f"Unknown model type: '{mtype}'. "
                     f"Choose from: ae | vae | graph_ae | simple_ae")


def build_loss(cfg):
    mtype          = cfg["model"]["type"]
    beta           = cfg["training"].get("beta", 1.0)
    density_weight = cfg["training"].get("density_weight", 0.05)

    if mtype == "vae":
        def loss_fn(output, x):
            recon, mu, log_var = output
            total, _, _ = vae_loss(recon, x, mu, log_var,
                                   beta=beta, density_weight=density_weight)
            return total
    else:
        def loss_fn(output, x):
            recon = output[0] if isinstance(output, tuple) else output
            return combined_ae_loss(recon, x, density_weight=density_weight)

    return loss_fn


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Train 3D anomaly detection model")
    parser.add_argument("--config",  default=str(root / "experiments/configs/default.yaml"))
    parser.add_argument("--resume",  choices=["last", "best"], default=None,
                        help="Resume from 'last.pt' or 'best.pt' checkpoint")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = MVTec3DDataset(
        root=str(root / cfg["dataset"]["root"]),
        category=cfg["dataset"]["category"],
        split="train",
        n_points=cfg["dataset"]["n_points"],
        augment=cfg["dataset"]["augment"],
    )

    val_size   = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=cfg["training"]["batch_size"],
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=cfg["training"]["batch_size"],
                              shuffle=False, num_workers=0, pin_memory=True)

    # ── Model & optimiser ─────────────────────────────────────────────────────
    model     = build_model(cfg)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
        eta_min=1e-5,
    )

    loss_fn = build_loss(cfg)

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=str(root / "experiments/checkpoints"),
        patience=cfg["training"].get("patience", 20),
    )

    resume_file = f"{args.resume}.pt" if args.resume else None
    trainer.fit(train_loader, cfg["training"]["epochs"], val_loader,
                resume_from=resume_file)

    print(f"\nTraining complete. Best checkpoint: experiments/checkpoints/best.pt")
    print(f"Model type  : {cfg['model']['type']}")
    print(f"Latent dim  : {cfg['model']['latent_dim']}")
    print(f"Category    : {cfg['dataset']['category']}")


if __name__ == "__main__":
    main()
