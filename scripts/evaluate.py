"""Entry point: evaluate model and compute anomaly scores."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve

from src.utils.seed import set_seed
from src.utils.dataset import MVTec3DDataset
from src.evaluation.anomaly_scorer import reconstruction_score
from src.evaluation.metrics import compute_auroc, compute_ap
from src.visualization.visualize import plot_roc_curve
from scripts.train import build_model


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(root / "experiments/configs/default.yaml"))
    parser.add_argument("--checkpoint", default=str(root / "experiments/checkpoints/last.pt"))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = cfg["device"]

    test_set = MVTec3DDataset(
        root=str(root / cfg["dataset"]["root"]),
        category=cfg["dataset"]["category"],
        split="test",
        n_points=cfg["dataset"]["n_points"],
    )
    loader = DataLoader(test_set, batch_size=1)

    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)

    all_scores, all_labels = [], []
    for x, label in loader:
        score = reconstruction_score(model, x, device=device)
        all_scores.append(score[0])
        all_labels.append(label.item())

    scores = np.array(all_scores)
    labels = np.array(all_labels)

    auc = compute_auroc(labels, scores)
    ap = compute_ap(labels, scores)
    print(f"AUROC: {auc:.4f}  |  AP: {ap:.4f}")

    fpr, tpr, _ = roc_curve(labels, scores)
    plot_roc_curve(fpr, tpr, auc, save_path="experiments/results/roc_curve.png")


if __name__ == "__main__":
    main()
