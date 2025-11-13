import torch
import torch.nn as nn

class EmbeddingEEGNet(nn.Module):
    """A PyTorch implementation of EEGNet for generating embeddings.

    This model is a convolutional neural network designed for EEG signal
    processing. It has been adapted from the original EEGNet architecture
to
    output a fixed-size embedding vector instead of class probabilities.

    Parameters
    ----------
    channels : int, optional
        Number of EEG channels, by default 4.
    timesteps : int, optional
        Number of time points in each EEG sample, by default 751.
    cnn_temp_filt : int, optional
        Number of filters in the initial temporal convolution, by default 8.
    cnn_temp_size : int, optional
        Kernel size of the initial temporal convolution, by default 64.
    cnn_separable_size : int, optional
        Kernel size of the separable convolution, by default 16. Also defines
        the number of output filters for this layer.
    depth : int, optional
        Depth multiplier for the depthwise convolution, by default 2.
    pool1s : int, optional
        Downsampling factor for the first average pooling layer, by default 4.
    pool2s : int, optional
        Downsampling factor for the second average pooling layer, by default 8.
    dropout1 : float, optional
        Dropout rate after the first pooling layer, by default 0.5.
    dropout2 : float, optional
        Dropout rate after the second pooling layer, by default 0.5.
    embedd_dim : int, optional
        The dimensionality of the final output embedding vector, by default 32.

    Attributes
    ----------
    cnnTemp : nn.Conv2d
        Temporal convolutional layer.
    batchNorm1 : nn.BatchNorm2d
        Batch normalization after the temporal convolution.
    cnnDepth : nn.Conv2d
        Depthwise convolutional layer.
    batchNorm2 : nn.BatchNorm2d
        Batch normalization after the depthwise convolution.
    avgPool1 : nn.AvgPool2d
        First average pooling layer.
    dropout1 : nn.Dropout
        First dropout layer.
    cnnSeparable : nn.Conv2d
        Separable convolutional layer.
    batchNorm3 : nn.BatchNorm2d
        Batch normalization after the separable convolution.
    avgPool2 : nn.AvgPool2d
        Second average pooling layer.
    dropout2 : nn.Dropout
        Second dropout layer.
    flatten : nn.Flatten
        Flattens the tensor for the fully connected layer.
    fc1 : nn.Linear
        Fully connected layer that produces the final embedding.
    elu : nn.ELU
        ELU activation function.
    """
    def __init__(self,
                 channels=4,
                 timesteps=751,
                 cnn_temp_filt=8,
                 cnn_temp_size=64,
                 cnn_separable_size=16,
                 depth=2,
                 pool1s=4,
                 pool2s=8,
                 dropout1=0.5,
                 dropout2=0.5,
                 maxnorm_depth=1,
                 embedd_dim=32
                ):
        super(EmbeddingEEGNet, self).__init__()

        self.cnnTemp = nn.Conv2d(
            in_channels=1,
            out_channels=cnn_temp_filt,
            kernel_size=(1, cnn_temp_size),
            padding="same"
        )

        self.batchNorm1 = nn.BatchNorm2d(cnn_temp_filt)

        self.cnnDepth = nn.Conv2d(
            in_channels=cnn_temp_filt,
            out_channels=depth*cnn_temp_filt,
            kernel_size=(depth,1),
            padding="valid"
        )

        self.batchNorm2 = nn.BatchNorm2d(depth * cnn_temp_filt)

        self.avgPool1 = nn.AvgPool2d((1,pool1s))

        self.dropout1 = nn.Dropout(dropout1)

        self.cnnSeparable = nn.Conv2d(
            in_channels=depth*cnn_temp_filt,
            out_channels=cnn_separable_size,
            kernel_size=(1,cnn_separable_size),
            padding="same"
        )

        self.batchNorm3 = nn.BatchNorm2d(cnn_separable_size)



        self.avgPool2 = nn.AvgPool2d((1,pool2s))
        self.dropout2 = nn.Dropout(dropout2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(
            in_features=(cnn_separable_size * (channels - depth + 1) * (timesteps // (pool1s * pool2s))),
            out_features=embedd_dim)

        self.elu = nn.ELU()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the EmbeddingEEGNet.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, 1, C, T), where N is the batch size,
            C is the number of channels, and T is the number of timesteps.

        Returns
        -------
        torch.Tensor
            The output embedding tensor of shape (N, embedd_dim).
        """
        x = self.cnnTemp(x)
        x = self.batchNorm1(x)
        x = self.cnnDepth(x)
        x = self.batchNorm2(x)
        x = self.elu(x)
        x = self.avgPool1(x)
        x = self.dropout1(x)
        x = self.cnnSeparable(x)
        x = self.batchNorm3(x)
        x = self.elu(x)
        x = self.avgPool2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        embedding = self.fc1(x)
        return embedding