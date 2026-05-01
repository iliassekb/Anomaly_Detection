"""PointNet feature extractor."""
import torch
import torch.nn as nn


class PointNetEncoder(nn.Module):
    """Local + global feature extraction from raw point clouds."""

    def __init__(self, global_feat: bool = True, feature_transform: bool = False):
        super().__init__()
        self.global_feat = global_feat
        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 3) → transpose to (B, 3, N)
        x = x.transpose(2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.max(dim=2).values  # global max pool → (B, 1024)
        return x


class PointNetClassifier(nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.encoder = PointNetEncoder()
        self.head = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        return self.head(feat)
