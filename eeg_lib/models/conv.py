import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


class EEGEmbedder(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super(EEGEmbedder, self).__init__()

        self.cnn = nn.Sequential(
            # Temporal convolution (within each channel)
            nn.Conv2d(
                1, 16, kernel_size=(1, 25), stride=(1, 1), padding=(0, 12), bias=False
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # Spatial convolution (across EEG channels)
            nn.Conv2d(16, 32, kernel_size=(4, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.25),
            # Further temporal processing
            nn.Conv2d(
                32, 64, kernel_size=(1, 15), stride=(1, 2), padding=(0, 7), bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # (B, C, 1, 1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, embedding_dim),
            nn.Tanh(),  # or ReLU, depending on use
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, 751)
        x = x.unsqueeze(1)  # -> (B, 1, 4, 751)
        x = self.cnn(x)
        x = self.global_pool(x)  # -> (B, 64, 1, 1)
        x = self.fc(x)  # -> (B, embedding_dim)
        return x
