import torch
import torch.nn as nn
import torch.nn.functional as F


class FBCNet(nn.Module):
    """
    A simplified implementation of FBCNet:
    - In FBCNet the EEG signal is pre-filtered into multiple frequency bands.
    - Here, we simulate band separation by replicating the EEG input along a new band dimension.
    - A group convolution is applied which mimics filtering using separate narrow-band filters.
    - Subsequent depthwise convolution, batch normalization, ELU, pooling and dropout are used.
    - A final FC layer produces a normalized embedding.
    """

    def __init__(self, num_channels, num_samples, embedding_size=32, num_bands=9):
        super(FBCNet, self).__init__()
        self.num_bands = num_bands
        # Simulated band-specific convolution: groups equal to num_bands
        self.conv1 = nn.Conv2d(1, 8 * num_bands, kernel_size=(1, 64), padding=(0, 32),
                               bias=False, groups=num_bands)
        self.depthwise_conv = nn.Conv2d(8 * num_bands, 16 * num_bands, kernel_size=(num_channels, 1),
                                        groups=8 * num_bands, bias=False)
        self.batchnorm = nn.BatchNorm2d(16 * num_bands)
        self.elu = nn.ELU()
        self.avgpool = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout = nn.Dropout(0.25)

        self.feature_dim = self._get_flattened_size(num_channels, num_samples)
        self.fc = nn.Linear(self.feature_dim, embedding_size)

    def _get_flattened_size(self, num_channels, num_samples):
        with torch.no_grad():
            # Create a dummy input and simulate band stacking.
            dummy_input = torch.zeros(1, 1, num_channels, num_samples)
            # Replicate along a new dimension to simulate pre-filtered bands.
            dummy_input = dummy_input.repeat(1, self.num_bands, 1, 1)  # shape (1, num_bands, channels, samples)
            # Merge band dimension and channel dimension
            dummy_input = dummy_input.view(1, 1, self.num_bands * num_channels, num_samples)
            x = self.conv1(dummy_input)
            x = self.depthwise_conv(x)
            x = self.batchnorm(x)
            x = self.elu(x)
            x = self.avgpool(x)
            x = self.dropout(x)
            return x.view(1, -1).shape[1]

    def forward(self, x):
        # x shape: (B, channels, samples)
        batch_size = x.shape[0]
        # Simulate pre-filtered frequency bands by replicating along a new dimension
        # New shape: (B, num_bands, channels, samples)
        x = x.unsqueeze(1).repeat(1, self.num_bands, 1, 1)
        # Merge the band and channel dimensions: (B, 1, num_bands * channels, samples)
        x = x.view(batch_size, 1, self.num_bands * x.shape[2], x.shape[3])
        x = self.conv1(x)
        x = self.depthwise_conv(x)
        x = self.batchnorm(x)
        x = self.elu(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
