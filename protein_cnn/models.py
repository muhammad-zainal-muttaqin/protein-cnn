from __future__ import annotations

import torch
from torch import nn


class CNN1DTagger(nn.Module):
    def __init__(self, in_channels: int = 42, num_classes: int = 8, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(256, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN2DTagger(nn.Module):
    def __init__(self, num_classes: int = 8, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(7, 5), padding=(3, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 64, kernel_size=(7, 5), padding=(3, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 64, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=(1, 42))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(-1)


def build_model(model_name: str) -> nn.Module:
    if model_name == "cnn1d":
        return CNN1DTagger()
    if model_name == "cnn2d":
        return CNN2DTagger()
    raise ValueError(f"Unknown model: {model_name}")
