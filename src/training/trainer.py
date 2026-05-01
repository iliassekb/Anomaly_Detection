"""Generic training loop with LR scheduling and early stopping."""
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "experiments/checkpoints",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        patience: int = 10,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.scheduler = scheduler
        self.patience = patience

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for i, batch in enumerate(loader):
            x = batch[0].to(self.device)
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.loss_fn(output, x)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            print(f"  batch {i+1}/{len(loader)}  loss={loss.item():.4f}", end="\r", flush=True)
        print()
        return total_loss / len(loader)

    def fit(self, train_loader: DataLoader, n_epochs: int, val_loader: DataLoader = None, resume_from: str = None):
        best_val_loss = float("inf")
        epochs_no_improve = 0
        start_epoch = 1

        if resume_from:
            ckpt = self.load_checkpoint(resume_from)
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt["best_val_loss"]
            epochs_no_improve = ckpt["epochs_no_improve"]
            logger.info(f"Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

        for epoch in range(start_epoch, n_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            lr = self.optimizer.param_groups[0]["lr"]
            msg = f"Epoch {epoch}/{n_epochs} — train_loss: {train_loss:.4f}  lr: {lr:.6f}"

            if val_loader:
                val_loss = self.evaluate(val_loader)
                msg += f"  val_loss: {val_loss:.4f}"

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    self.save_checkpoint("best.pt", epoch, best_val_loss, epochs_no_improve)
                else:
                    epochs_no_improve += 1

            logger.info(msg)

            if self.scheduler:
                self.scheduler.step()

            self.save_checkpoint("last.pt", epoch, best_val_loss, epochs_no_improve)

            if epochs_no_improve >= self.patience:
                logger.info(f"Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                break

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        for batch in loader:
            x = batch[0].to(self.device)
            output = self.model(x)
            total_loss += self.loss_fn(output, x).item()
        return total_loss / len(loader)

    def save_checkpoint(self, name: str, epoch: int, best_val_loss: float, epochs_no_improve: int):
        path = self.checkpoint_dir / name
        torch.save({
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "epochs_no_improve": epochs_no_improve,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
        }, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, name: str) -> dict:
        path = self.checkpoint_dir / name
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if self.scheduler and ckpt.get("scheduler_state"):
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        logger.info(f"Checkpoint loaded: {path}")
        return ckpt
