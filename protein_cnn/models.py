from __future__ import annotations

import torch
from torch import nn


class CNN1DTagger(nn.Module):
    def __init__(
        self,
        in_channels: int = 42,
        num_classes: int = 8,
        dropout: float = 0.2,
        hidden_channels: tuple[int, int, int] = (128, 256, 256),
    ):
        super().__init__()
        c1, c2, c3 = hidden_channels
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=7, padding=3),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(c1, c2, kernel_size=7, padding=3),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(c2, c3, kernel_size=5, padding=2),
            nn.BatchNorm1d(c3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(c3, num_classes, kernel_size=1),
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


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ResidualDilatedCNN1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 42,
        num_classes: int = 8,
        dropout: float = 0.2,
        channels: int = 256,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 1, 2, 4, 8),
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock1D(channels=channels, dilation=d, dropout=dropout) for d in dilations]
        )
        self.head = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


def build_model(model_name: str, **kwargs) -> nn.Module:
    if model_name == "cnn1d":
        return CNN1DTagger(**kwargs)
    if model_name == "cnn2d":
        return CNN2DTagger(**kwargs)
    if model_name == "resdil_cnn1d":
        return ResidualDilatedCNN1D(**kwargs)
    raise ValueError(f"Unknown model: {model_name}")
