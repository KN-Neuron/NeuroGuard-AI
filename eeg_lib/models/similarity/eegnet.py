import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """
    A simplified implementation of EEGNet:
    - Temporal convolution with fixed kernel size to capture time-domain features.
    - Depthwise convolution (one filter per channel group) to model spatial relationships.
    - Batch normalization, ELU activation, average pooling and dropout.
    - Final fully connected layer outputs an embedding of given size.
    The output embedding is L2-normalized.
    """

    def __init__(self, num_channels, num_samples, embedding_size=32):
        super(EEGNet, self).__init__()
        # Ensure kernel size doesn't exceed channel count
        temporal_kernel_size = (1, 64)
        spatial_kernel_size = (num_channels, 1)  # Adaptive kernel size

        self.temporal_conv = nn.Conv2d(1, 8, kernel_size=temporal_kernel_size, padding=(0, 32), bias=False)
        self.depthwise_conv = nn.Conv2d(8, 16, kernel_size=spatial_kernel_size, groups=8, bias=False)
        self.batchnorm = nn.BatchNorm2d(16)
        self.elu = nn.ELU()
        self.avgpool = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout = nn.Dropout(0.25)

        # Determine the flattened feature size dynamically
        self.feature_dim = self._get_flattened_size(num_channels, num_samples)
        self.fc = nn.Linear(self.feature_dim, embedding_size)

    def _get_flattened_size(self, num_channels, num_samples):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, num_channels, num_samples)
            x = self.temporal_conv(dummy_input)
            x = self.depthwise_conv(x)
            x = self.batchnorm(x)
            x = self.elu(x)
            x = self.avgpool(x)
            x = self.dropout(x)
            return x.view(1, -1).shape[1]

    def forward(self, x):
        # x shape: (B, channels, samples)
        x = x.unsqueeze(1)  # new shape: (B, 1, channels, samples)
        x = self.temporal_conv(x)
        x = self.depthwise_conv(x)
        x = self.batchnorm(x)
        x = self.elu(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # L2 normalization makes the embedding lie on a unit hypersphere
        x = F.normalize(x, p=2, dim=1)
        return x