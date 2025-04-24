import torch
import torch.nn as nn


# Model Grzesia zeby sprawdic czy to problem w modelu faktycznie
class EEGNetEmbeddingModel(nn.Module):
    """
    EEGNet model adapted for generating embeddings from EEG data for verification tasks.

    This model is based on the EEGNet architecture (EEGNET-8,2 variant) and is modified to output
    a fixed-size embedding vector (default 32 dimensions) that can be used to compare EEG signals.
    It also includes a classification head for auxiliary training if desired.

    Attributes:
        temporal_conv_block (nn.Sequential): The block performing temporal convolution.
        spatial_conv_block (nn.Sequential): The block performing spatial (depthwise) convolution.
        separable_conv_block (nn.Sequential): The block performing separable convolution (separable + pointwise).
        flatten_layer (nn.Flatten): Flattens the convolution output.
        embedding_layer (nn.Linear): Linear layer to produce the final embedding.
        classification_layer (nn.Linear): Linear layer that maps embeddings to class logits.
    """

    def __init__(
        self,
        num_channels: int = 4,
        num_classes: int = 4,
        num_time_points: int = 751,
        temporal_kernel_size: int = 32,
        num_filters_first_layer: int = 16,
        num_filters_second_layer: int = 32,
        depth_multiplier: int = 2,
        pool_kernel_size_1: int = 8,
        pool_kernel_size_2: int = 16,
        dropout_rate: float = 0.5,
        max_norm_depthwise: float = 1.0,
        max_norm_linear: float = 0.25,
        embedding_dimension: int = 32,
    ):
        """
        Initialize the EEGNetEmbeddingModel.

        Args:
            num_channels (int): Number of EEG channels (default 4).
            num_classes (int): Number of output classes for classification (default 4).
            num_time_points (int): Number of time points per EEG epoch (default 751).
            temporal_kernel_size (int): Kernel size for temporal convolution (default 32).
            num_filters_first_layer (int): Number of filters for the first convolutional layer (default 16).
            num_filters_second_layer (int): Number of filters for the second convolutional block (default 32).
            depth_multiplier (int): Multiplier for depthwise convolution (default 2).
            pool_kernel_size_1 (int): Pooling kernel size in the first pooling layer (default 8).
            pool_kernel_size_2 (int): Pooling kernel size in the second pooling layer (default 16).
            dropout_rate (float): Dropout rate (default 0.5).
            max_norm_depthwise (float): Maximum norm value for depthwise convolution weights (default 1.0).
            max_norm_linear (float): Maximum norm value for linear layer weights (default 0.25).
            embedding_dimension (int): Dimensionality of the embedding vector (default 32).
        """
        super(EEGNetEmbeddingModel, self).__init__()

        # calc flattened size after convolution and pooling layers.
        flattened_size = (
            num_time_points // (pool_kernel_size_1 * pool_kernel_size_2)
        ) * num_filters_second_layer

        self.temporal_conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=num_filters_first_layer,
                kernel_size=(1, temporal_kernel_size),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_filters_first_layer),
        )

        self.spatial_conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters_first_layer,
                out_channels=depth_multiplier * num_filters_first_layer,
                kernel_size=(num_channels, 1),
                groups=num_filters_first_layer,
                bias=False,
            ),
            nn.BatchNorm2d(depth_multiplier * num_filters_first_layer),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool_kernel_size_1)),
            nn.Dropout(p=dropout_rate),
        )

        self.separable_conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=depth_multiplier * num_filters_first_layer,
                out_channels=num_filters_second_layer,
                kernel_size=(1, 16),
                groups=num_filters_second_layer,
                bias=False,
                padding="same",
            ),
            nn.Conv2d(
                in_channels=num_filters_second_layer,
                out_channels=num_filters_second_layer,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_filters_second_layer),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool_kernel_size_2)),
            nn.Dropout(p=dropout_rate),
        )

        self.flatten_layer = nn.Flatten()

        self.embedding_layer = nn.Linear(flattened_size, embedding_dimension)


        self.apply_max_norm_to_layer(self.spatial_conv_block[0], max_norm_depthwise)

    def apply_max_norm_to_layer(self, layer: nn.Module, max_norm_value: float):
        """
        Apply a max-norm constraint to the weights of a given layer.

        Args:
            layer (nn.Module): The layer to which the max-norm constraint will be applied.
            max_norm_value (float): The maximum allowed norm for the weights.
        """
        for name, param in layer.named_parameters():
            if "weight" in name:
                param.data = torch.renorm(
                    param.data, p=2, dim=0, maxnorm=max_norm_value
                )

    def forward(self, input_tensor: torch.Tensor):
        """
        Forward pass through the EEGNetEmbeddingModel.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape
                (batch_size, 1, num_channels, num_time_points).

        Returns:
            tuple:
                - embedding_vector (torch.Tensor): The output embedding vector of shape
                  (batch_size, embedding_dimension) for each input.
                - classification_logits (torch.Tensor): The classification logits of shape
                  (batch_size, num_classes) for each input.
        """
        x = self.temporal_conv_block(input_tensor)
        x = self.spatial_conv_block(x)
        x = self.separable_conv_block(x)
        x = self.flatten_layer(x)
        embedding_vector = self.embedding_layer(x)
        return embedding_vector