import torch
import torch.nn as nn

class EmbeddingEEGNet(nn.Module):
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

        self.avgPool1 = nn.AvgPool2d((1,pool1s))

        self.dropout1 = nn.Dropout(dropout1)
        self.cnnSeparable = nn.Conv2d(
            in_channels=depth*cnn_temp_filt,
            out_channels=cnn_separable_size,
            kernel_size=(1,cnn_separable_size),
            padding="same"
        )

        self.batchNorm2 = nn.BatchNorm2d(cnn_separable_size)

        self.avgPool2 = nn.AvgPool2d((1,pool2s))
        self.dropout2 = nn.Dropout(dropout2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=(depth * cnn_temp_filt * (channels - depth + 1) * (timesteps // (pool1s * pool2s))),
                             out_features=embedd_dim)

        self.elu = nn.ELU()


    def forward(self, x):
        x = self.cnnTemp(x)
        x = self.batchNorm1(x)
        x = self.cnnDepth(x)
        x = self.batchNorm2(x)
        x = self.elu(x)
        x = self.avgPool1(x)
        x = self.dropout1(x)
        x = self.cnnSeparable(x)
        x = self.batchNorm2(x)
        x = self.elu(x)
        x = self.avgPool2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        embedding = self.fc1(x)
        return embedding