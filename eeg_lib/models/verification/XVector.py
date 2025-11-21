import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
from torch import optim
from eeg_lib.losses.proxynca_loss import ProxyNCALoss
import torch.nn.functional as F
import math
from eeg_lib.losses.arcface_loss import ArcMarginProduct
from typing import Optional, Dict, Any, Union, Tuple
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np


class StatisticsPooling(nn.Module):
    """Computes mean and standard deviation across the temporal dimension.

    This layer takes a batch of sequences as input and computes the mean and
    standard deviation for each sequence over the time dimension (dim=2).
    The resulting statistics are then concatenated to form the output.

    Parameters
    ----------
    eps : float, optional
        A small value added to the variance for numerical stability before
        taking the square root, by default 1e-6.

    Attributes
    ----------
    eps : float
        The epsilon value for numerical stability.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super(StatisticsPooling, self).__init__()
        self.eps: float = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass for the StatisticsPooling layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, D, T), where N is the batch size,
            D is the feature dimension, and T is the number of timesteps.

        Returns
        -------
        torch.Tensor
            The output tensor of shape (N, 2*D), where the mean and
            standard deviation are concatenated along the feature dimension.
        """
        mean = torch.mean(x, dim=2)
        std = torch.sqrt(torch.var(x, dim=2, unbiased=False) + self.eps)
        pooled = torch.cat((mean, std), dim=1)
        return pooled


class EcapaSEModule(nn.Module):
    """
    Squeeze and excitation module used in ECAPA_TDNN model and in EcapaSEResBlock

    Parameters
    ----------
    in_channels : int
        The number of channels in the input tensor.
    reduction : int, optional
        The reduction factor for the dimensionality of the intermediate
        bottleneck layer in the excitation phase, by default 8.

    Attributes
    ----------
    squeeze : nn.AdaptiveAvgPool1d
        Squeezes the temporal dimension to produce a channel descriptor.
    excitation : nn.Sequential
        A two-layer neural network that computes the channel-wise attention
        weights (excitations).
    """

    def __init__(self, in_channels: int, reduction: int = 8) -> None:

        super(EcapaSEModule, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        self.squeeze = nn.AdaptiveAvgPool1d(1)

        self.excitation = nn.Sequential(  # Squeeze, Dense1, Relu, Dense2, Sigmoid
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels // reduction,
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=in_channels // reduction,
                out_channels=in_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the Squeeze-and-Excitation module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, L), where N is the batch size, C is
            the number of channels (`in_channels`), and L is the sequence length.

        Returns
        -------
        torch.Tensor
            The output tensor with recalibrated channel-wise features, having the
            same shape as the input tensor (N, C, L).
        """
        s = self.squeeze(x)
        e = self.excitation(s)
        return e * x


class EcapaSEResBlock(nn.Module):
    """
    Res2Net block and squeeze-excitation integrated together, used in ECAPA_TDNN model,
    this is the building block of the standard ECAPA_TDNN

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    split : int, optional
        The number of feature splits for the Res2Net module, by default 4.
    kernel_size : int, optional
        Kernel size for the convolutions within the Res2Net module, by default 3.
    padding : int, optional
        This parameter is defined for compatibility but padding is calculated
        dynamically based on dilation and kernel_size, by default 1.
    dilation : int, optional
        Dilation factor for the convolutions within the Res2Net module, by default 0.

    Attributes
    ----------
    conv1 : nn.Conv1d
        Initial 1x1 convolution to change channel dimensions.
    relu : nn.ReLU
        ReLU activation function.
    bn1 : nn.GroupNorm
        Group normalization after the first convolution.
    width : int
        The number of channels in each split.
    split : int
        The number of feature splits.
    convs : nn.ModuleList
        List of convolutional layers for each split in the Res2Net block.
    bns : nn.ModuleList
        List of group normalization layers for each split.
    conv2 : nn.Conv1d
        Final 1x1 convolution after concatenating the splits.
    bn2 : nn.GroupNorm
        Group normalization after the second convolution.
    se : EcapaSEModule
        Squeeze-and-Excitation module for channel attention.
    shortcut : nn.Module
        Shortcut connection (1x1 convolution or Identity) for the residual link.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        split: int = 4,
        kernel_size: int = 3,
        padding: int = 1,
        dilation: int = 0,
    ) -> None:
        super(EcapaSEResBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.GroupNorm(1, out_channels)

        self.width = out_channels // split
        self.split = split
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    self.width,
                    self.width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=(dilation * (kernel_size - 1)) // 2,
                )
                for _ in range(split - 1)
            ]
        )

        self.bns = nn.ModuleList(
            [nn.GroupNorm(1, self.width) for _ in range(split - 1)]
        )
        self.conv2 = nn.Conv1d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1
        )
        self.bn2 = nn.GroupNorm(1, out_channels)

        self.se = EcapaSEModule(out_channels)

        self.shortcut = (
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, no_norm: bool = False) -> torch.Tensor:
        """Defines the forward pass for the EcapaSEResBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C_in, L), where N is the batch size,
            C_in is `in_channels`, and L is the sequence length.
        no_norm : bool, optional
            If True, bypasses the group normalization layers, by default False.

        Returns
        -------
        torch.Tensor
            The output tensor of shape (N, C_out, L), where C_out is `out_channels`.
        """
        residual = self.shortcut(x)
        x = self.relu(self.conv1(x))
        if not no_norm:
            x = self.bn1(x)

        frames = torch.split(x, self.width, 1)
        outputs = [frames[0]]
        y = frames[0]

        for i in range(len(self.convs)):
            y_conv = self.convs[i](y)
            if not no_norm:
                y_conv = self.bns[i](y_conv)
            y_conv = self.relu(y_conv)

            y = y_conv + frames[i + 1]
            outputs.append(y)
        x = torch.cat(outputs, 1)

        x = self.conv2(x)
        if not no_norm:
            x = self.bn2(x)
        x = self.relu(x)

        x = self.se(x)
        return self.relu(x + residual)


class AttentiveStatisticsPooling(nn.Module):
    """
    Final layer in the ECAPA_TDNN architecture aggregates statistics across timesteps
    and scales them by the attention factor

    Parameters
    ----------
    in_channels : int
        The number of channels or feature dimensions in the input tensor.
    eps : float, optional
        A small value added to the variance for numerical stability before
        taking the square root, by default 1e-6.
    reduction : int, optional
        The dimensionality of the hidden layer in the attention network,
        by default 128.

    Attributes
    ----------
    eps : float
        The epsilon value for numerical stability.
    att : nn.Sequential
        The attention network that computes the weights for each time step.
    """

    def __init__(
        self, in_channels: int, eps: float = 1e-6, reduction: int = 128
    ) -> None:
        super(AttentiveStatisticsPooling, self).__init__()
        self.in_channels = in_channels
        self.eps = eps
        self.reduction = reduction
        self.att = nn.Sequential(
            nn.Linear(in_channels, reduction), nn.ReLU(), nn.Linear(reduction, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass for the AttentiveStatisticsPooling layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, T), where N is the batch size,
            C is the number of channels (`in_channels`), and T is the
            number of time steps.

        Returns
        -------
        torch.Tensor
            The aggregated utterance-level embedding of shape (N, 2*C),
            containing the concatenated weighted mean and standard deviation.
        """

        batch_size, num_channels, time_steps = x.shape
        frames = x.transpose(1, 2)
        scores = self.att(frames)
        weights = F.softmax(scores.squeeze(-1), dim=1)
        weights = weights.unsqueeze(-1)

        weighted_sum = torch.sum(weights * frames, dim=1)
        sqr_frames = frames * frames
        weighted_sqr_sum = torch.sum(weights * sqr_frames, dim=1)
        variance = weighted_sqr_sum - weighted_sum * weighted_sum
        stddev = torch.sqrt(torch.clamp(variance, min=self.eps))
        pooled = torch.cat((weighted_sum, stddev), dim=1)
        return pooled


class ECAPA_TDNN(nn.Module):
    """
    Standard ECAPA_TDNN model with SE and Res2Net blocks integrated together

    Parameters
    ----------
    input_features : int
        The number of input features
    num_classes : int, optional
        The number of classes for the output classifier. If None, the model
        will only output embeddings, by default None.
    embedding_dim : int, optional
        The dimensionality of the final output embedding, by default 32.
    layer1_filt : int, optional
        Number of filters in the first EcapaSEResBlock, by default 512.
    layer2_filt : int, optional
        Number of filters in the second EcapaSEResBlock, by default 512.
    layer3_filt : int, optional
        Number of filters in the third EcapaSEResBlock, by default 1024.
    layer4_filt : int, optional
        Number of filters in the fourth EcapaSEResBlock, by default 1024.
    dropout1 : float, optional
        Dropout rate after the first SE-Res2Net block, by default 0.25.
    dropout2 : float, optional
        Dropout rate after the second SE-Res2Net block, by default 0.25.
    dropout3 : float, optional
        Dropout rate after the third SE-Res2Net block, by default 0.25.

    Attributes
    ----------
    layer1, layer2, layer3, layer4 : EcapaSEResBlock
        The SE-Res2Net blocks of the network.
    cat_conv : nn.Conv1d
        1x1 convolution layer applied after concatenating multi-scale features.
    pooling : AttentiveStatisticsPooling
        Attentive statistics pooling layer.
    ln : nn.GroupNorm
        Group normalization layer.
    dense : nn.Linear
        A fully connected layer before the final embedding layer.
    embeddingLayer : nn.Linear
        The final fully connected layer that produces the embedding.
    classifier : nn.Linear or None
        The output classifier layer. None if `num_classes` is not specified.
    """

    def __init__(
        self,
        input_features: int,
        num_classes: Optional[int] = None,
        embedding_dim: int = 32,
        layer1_filt: int = 512,
        layer2_filt: int = 512,
        layer3_filt: int = 1024,
        layer4_filt: int = 1024,
        layer5_filt: int = 1500,
        dropout1: float = 0.25,
        dropout2: float = 0.25,
        dropout3: float = 0.25,
        dropout4: float = 0.25,
    ) -> None:
        super().__init__()

        self.relu = nn.ReLU()

        self.layer1 = EcapaSEResBlock(
            in_channels=input_features,
            out_channels=layer1_filt,
            kernel_size=3,
            dilation=2,
        )
        self.dropout1 = nn.Dropout(dropout1)
        self.layer2 = EcapaSEResBlock(
            in_channels=layer1_filt, out_channels=layer2_filt, kernel_size=3, dilation=3
        )
        self.dropout2 = nn.Dropout(dropout2)
        self.layer3 = EcapaSEResBlock(
            in_channels=layer2_filt, out_channels=layer3_filt, kernel_size=3, dilation=4
        )

        self.dropout3 = nn.Dropout(dropout3)

        self.layer4 = EcapaSEResBlock(
            in_channels=layer3_filt, out_channels=layer4_filt, kernel_size=3, dilation=5
        )

        self.cat_conv = nn.Conv1d(
            layer1_filt + layer2_filt + layer3_filt + layer4_filt, 1500, kernel_size=1
        )
        self.pooling = AttentiveStatisticsPooling(in_channels=1500)

        self.ln = nn.GroupNorm(1, 3000)
        self.dense = nn.Linear(in_features=3000, out_features=1024)
        self.embeddingLayer = nn.Linear(in_features=1024, out_features=embedding_dim)
        if num_classes is not None:
            self.classifier = nn.Linear(
                in_features=embedding_dim, out_features=num_classes
            )
        else:
            self.classifier = None

    def forward(
        self, x: torch.Tensor, return_embedding: bool = False, no_norm: bool = True
    ) -> torch.Tensor:
        """Defines the forward pass of the ECAPA-TDNN model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C_in, T), where N is the batch size,
            C_in is `input_features`, and T is the number of time steps.
        return_embedding : bool, optional
            If True, the model returns the final embedding. Otherwise, it
            returns the output of the classifier (if it exists), by default False.
        no_norm : bool, optional
            If True, bypasses the group normalization layer before the dense
            layers, by default True.

        Returns
        -------
        torch.Tensor
            The output tensor. This can be the embedding vector (N, `embedding_dim`)
            or the class logits (N, `num_classes`).
        """
        out1 = self.layer1(x)
        x = self.dropout1(out1)
        out2 = self.layer2(x)
        x = self.dropout2(out2)
        out3 = self.layer3(x)
        x = self.dropout3(out3)
        out4 = self.layer4(x)

        x_cat = torch.cat((out1, out2, out3, out4), dim=1)
        x = self.relu(self.cat_conv(x_cat))

        x = self.pooling(x)
        if not no_norm:
            x = self.ln(x)
        x = self.dense(x)
        x = self.relu(x)
        x = self.embeddingLayer(x)
        x = F.normalize(x, p=2, dim=1)
        if return_embedding or self.classifier is None:
            return x
        else:
            x = self.classifier(x)
            return x


class XVectorEmbeddingModel(nn.Module):
    """
    Standard TDNN Model

    Parameters
    ----------
    input_features : int
        The number of input features
    num_classes : int, optional
        The number of classes for the final classifier. If None, the model
        only outputs embeddings, by default None.
    embedding_dim : int, optional
        The dimensionality of the output embedding, by default 32.
    layer1_filt, layer2_filt, layer3_filt, layer4_filt, layer5_filt : int, optional
        Number of filters for each of the five frame layers.
    layer_1_dilatation, layer_2_dilatation, layer_3_dilatation : int, optional
        Dilation factors for the first three frame layers.
    layer_1_stride, layer_2_stride, layer_3_stride : int, optional
        Stride values for the first three frame layers.
    dropout1, dropout2, dropout3, dropout4 : float, optional
        Dropout rates applied after the first four frame layers.

    Attributes
    ----------
    Frame1, Frame2, Frame3, Frame4, Frame5 : nn.Conv1d
        The 1D convolutional frame layers.
    bn1, bn2, bn3, bn4, bn5 : nn.LayerNorm
        Layer normalization applied after each frame layer.
    dropout1, dropout2, dropout3, dropout4 : nn.Dropout
        Dropout layers.
    statPooling : StatisticsPooling
        Statistics pooling layer to aggregate features over time.
    embeddingLayer : nn.Linear
        The final fully connected layer that produces the embedding.
    classifier : nn.Linear or None
        The output classifier layer. None if `num_classes` is not specified.
    """

    def __init__(
        self,
        input_features: int,
        num_classes: Optional[int] = None,
        timesteps: int = 751,
        embedding_dim: int = 32,
        layer1_filt: int = 512,
        layer2_filt: int = 512,
        layer3_filt: int = 1024,
        layer4_filt: int = 1024,
        layer5_filt: int = 1500,
        layer_1_dilatation: int = 1,
        layer_2_dilatation: int = 2,
        layer_3_dilatation: int = 3,
        layer_1_stride: int = 1,
        layer_2_stride: int = 1,
        layer_3_stride: int = 2,
        dropout1: float = 0.25,
        dropout2: float = 0.25,
        dropout3: float = 0.25,
        dropout4: float = 0.25,
    ) -> None:

        super(XVectorEmbeddingModel, self).__init__()

        self.relu = nn.ReLU()
        self.Frame1 = nn.Conv1d(
            in_channels=input_features,
            out_channels=layer1_filt,
            kernel_size=5,
            stride=layer_1_stride,
            dilation=layer_1_dilatation,
            padding=2,
        )
        self.bn1 = nn.LayerNorm(layer1_filt)
        self.dropout1 = nn.Dropout(p=dropout1)
        self.Frame2 = nn.Conv1d(
            in_channels=layer1_filt,
            out_channels=layer2_filt,
            kernel_size=3,
            stride=layer_2_stride,
            dilation=layer_2_dilatation,
            padding=2,
        )
        self.bn2 = nn.LayerNorm(layer2_filt)

        self.dropout2 = nn.Dropout(p=dropout2)

        self.Frame3 = nn.Conv1d(
            in_channels=layer2_filt,
            out_channels=layer3_filt,
            kernel_size=3,
            stride=layer_3_stride,
            dilation=layer_3_dilatation,
            padding=3,
        )
        self.bn3 = nn.LayerNorm(layer3_filt)

        self.dropout3 = nn.Dropout(p=dropout3)

        self.Frame4 = nn.Conv1d(
            in_channels=layer3_filt,
            out_channels=layer4_filt,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.bn4 = nn.LayerNorm(layer4_filt)

        self.dropout4 = nn.Dropout(p=dropout4)

        self.Frame5 = nn.Conv1d(
            in_channels=layer4_filt,
            out_channels=layer5_filt,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.bn5 = nn.LayerNorm(layer5_filt)

        self.statPooling = StatisticsPooling(eps=1e-6)

        self.embeddingLayer = nn.Linear(
            in_features=layer5_filt * 2, out_features=embedding_dim
        )
        if num_classes is not None:
            self.classifier = nn.Linear(embedding_dim, num_classes)
        else:
            self.classifier = None

    def forward(
        self, x: torch.Tensor, return_embedding: bool = False, no_norm: bool = False
    ) -> torch.Tensor:
        """Defines the forward pass of the X-Vector TDNN model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C_in, T), where N is the batch size,
            C_in is `input_features`, and T is the number of time steps.
        return_embedding : bool, optional
            If True, returns the embedding. Otherwise, returns the output of
            the classifier (if it exists), by default False.
        no_norm : bool, optional
            If True, bypasses the layer normalization steps, by default False.

        Returns
        -------
        torch.Tensor
            The output tensor, which can be the embedding (N, `embedding_dim`)
            or the class logits (N, `num_classes`).
        """
        x = self.Frame1(x)
        if not no_norm:
            x = x.permute(0, 2, 1)
            x = self.bn1(x)
            x = x.permute(0, 2, 1)
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.Frame2(x)
        if not no_norm:
            x = x.permute(0, 2, 1)
            x = self.bn2(x)
            x = x.permute(0, 2, 1)
        x = self.dropout2(x)
        x = self.relu(x)
        x = self.Frame3(x)
        if not no_norm:
            x = x.permute(0, 2, 1)
            x = self.bn3(x)
            x = x.permute(0, 2, 1)
        x = self.dropout3(x)
        x = self.relu(x)
        x = self.Frame4(x)
        if not no_norm:
            x = x.permute(0, 2, 1)
            x = self.bn4(x)
            x = x.permute(0, 2, 1)
        x = self.dropout4(x)
        x = self.relu(x)
        x = self.Frame5(x)
        if not no_norm:
            x = x.permute(0, 2, 1)
            x = self.bn5(x)
            x = x.permute(0, 2, 1)
        x = self.relu(x)
        x = self.statPooling(x)
        x = self.embeddingLayer(x)
        if return_embedding or self.classifier is None:
            return x
        else:
            return self.classifier(x)


def get_standard_model(
    hparams: Dict[str, Any], input_features: int, num_classes: Optional[int]
) -> XVectorEmbeddingModel:
    """
    Utility function for creating standard TDNN model with hparams dictionary, instead of writing it all explicitly

    Parameters
    ----------
    hparams : dict[str, any]
        A dictionary containing the hyperparameters for the model. Expected keys are:
        "embedding_dim" (int), "dropout_rate" (float), "layer1_filters" (int),
        "layer2_filters" (int), "layer3_filters" (int), "layer4_filters" (int),
        "layer5_filters" (int), "layer_1_dilatation" (int), "layer_2_dilatation" (int),
        "layer_3_dilatation" (int), "layer_1_stride" (int), "layer_2_stride" (int),
        "layer_3_stride" (int).
    input_features : int
        The number of input features for the model
    num_classes : int or None
        The number of output classes for the final classifier. If None, the model
        will be configured to output embeddings only.

    Returns
    -------
    XVectorEmbeddingModel
        An instance of the `XVectorEmbeddingModel` initialized with the
        specified hyperparameters.
    """

    embedding_dim = hparams["embedding_dim"]
    dropout_rate = hparams["dropout_rate"]

    model = XVectorEmbeddingModel(
        input_features=input_features,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        dropout1=dropout_rate,
        dropout2=dropout_rate,
        dropout3=dropout_rate,
        dropout4=dropout_rate,
        layer1_filt=hparams["layer1_filters"],
        layer2_filt=hparams["layer2_filters"],
        layer3_filt=hparams["layer3_filters"],
        layer4_filt=hparams["layer4_filters"],
        layer5_filt=hparams["layer5_filters"],
        layer_1_dilatation=hparams["layer_1_dilatation"],
        layer_2_dilatation=hparams["layer_2_dilatation"],
        layer_3_dilatation=hparams["layer_3_dilatation"],
        layer_1_stride=hparams["layer_1_stride"],
        layer_2_stride=hparams["layer_2_stride"],
        layer_3_stride=hparams["layer_3_stride"],
    )
    return model


def get_ecapa_model(
    hparams: Dict[str, Any], input_features: int, num_classes: Optional[int]
) -> ECAPA_TDNN:
    """
    Utility function for creating ECAPA_TDNN model with hparams dictionary, instead of writing it all explicitly

    Parameters
    ----------
    hparams : dict[str, any]
        A dictionary containing the hyperparameters for the model. Expected keys are:
        "embedding_dim" (int), "dropout_rate" (float), "layer1_filters" (int),
        "layer2_filters" (int), "layer3_filters" (int), "layer4_filters" (int),
        "layer5_filters" (int).
    input_features : int
        The number of input features for the model (e.g., MFCC dimensions).
    num_classes : int or None
        The number of output classes for the final classifier. If None, the model
        will be configured to output embeddings only.

    Returns
    -------
    ECAPA_TDNN
        An instance of the `ECAPA_TDNN` model initialized with the
        specified hyperparameters.
    """

    embedding_dim = hparams["embedding_dim"]
    dropout_rate = hparams["dropout_rate"]

    model = ECAPA_TDNN(
        input_features=input_features,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        dropout1=dropout_rate,
        dropout2=dropout_rate,
        dropout3=dropout_rate,
        dropout4=dropout_rate,
        layer1_filt=hparams["layer1_filters"],
        layer2_filt=hparams["layer2_filters"],
        layer3_filt=hparams["layer3_filters"],
        layer4_filt=hparams["layer4_filters"],
        layer5_filt=hparams["layer5_filters"],
    )
    return model


def pretrain(
    hparams: Dict[str, Any],
    device: Union[torch.device, str],
    input_features: int,
    num_classes: int,
    dataloader: DataLoader,
    writer: Optional[SummaryWriter] = None,
    type: str = "standard",
    fold: Union[str, int] = "",
) -> nn.Module:
    """Creates and pretrains a model using a softmax cross-entropy loss.

    Parameters
    ----------
    hparams : dict[str, any]
        A dictionary of hyperparameters. Expected keys include:
        "softmax_learning_rate" (float), "softmax_learning_rate_decay" (float),
        and "softmax_epochs" (int), plus model-specific parameters.
    device : torch.device or str
        The device (e.g., "cpu" or "cuda") on which to train the model.
    input_features : int
        The dimensionality of the input features.
    num_classes : int
        The number of target classes for the classification task.
    dataloader : torch.utils.data.DataLoader
        The data loader for training data.
    writer : torch.utils.tensorboard.SummaryWriter, optional
        An optional SummaryWriter for logging metrics to TensorBoard, by default None.
    type : str, optional
        The type of model to train. "standard" for X-Vector TDNN, or "ecapa"
        for ECAPA-TDNN, by default "standard".
    fold : str or int, optional
        An identifier for the current fold in cross-validation, used for
        logging purposes, by default "".

    Returns
    -------
    torch.nn.Module
        The trained model.
    """

    lr = hparams["softmax_learning_rate"]
    weight_decay = hparams["softmax_learning_rate_decay"]
    epochs = hparams["softmax_epochs"]

    if type == "standard":
        model = get_standard_model(hparams, input_features, num_classes).to(device)
    else:
        model = get_ecapa_model(hparams, input_features, num_classes).to(device)
    if writer is not None and (fold == "" or fold == 1):
        rnd_sample = torch.randn(1, input_features, num_classes).to(device)
        writer.add_graph(model, rnd_sample)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=weight_decay)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        total_seen = 0

        for data, labels in dataloader:
            data = data.to(device)
            labels = labels.to(device).long()
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            pred = logits.argmax(dim=1)
            total_correct += (pred == labels).sum().item()
            total_seen += data.size(0)

        avg_loss = total_loss / total_seen
        acc = total_correct / total_seen
        if writer is not None:
            writer.add_scalar(f"Pretrain/Loss{fold}", avg_loss, epoch)
            writer.add_scalar(f"Pretrain/Accuracy{fold}", acc, epoch)
        print(
            f"[Pretrain] Epoch {epoch + 1}/{epochs}  Loss={avg_loss:.4f}  Acc={acc:.4f}"
        )
        scheduler.step()
    return model


def fine_tune(
    model: Union[ECAPA_TDNN, XVectorEmbeddingModel],
    hparams: Dict[str, Any],
    device: Union[torch.device, str],
    dataloader: DataLoader,
    num_classes: int,
    writer: Optional[SummaryWriter] = None,
    fold: Union[str, int] = "",
) -> Union[ECAPA_TDNN, XVectorEmbeddingModel]:
    """
    Function training a model with ProxyNCALoss

    Parameters
    ----------
    model : nn.Module
        The model to be fine-tuned.
    hparams : dict[str, any]
        A dictionary of hyperparameters. Expected keys include:
        "proxy_learning_rate" (float), "proxy_learning_rate_decay" (float),
        "proxy_epochs" (int), "scale" (float), and "embedding_dim" (int).
    device : torch.device or str
        The device (e.g., "cpu" or "cuda") on which to train the model.
    dataloader : torch.utils.data.DataLoader
        The data loader for the fine-tuning data.
    num_classes : int
        The number of classes for the ProxyNCA loss.
    writer : torch.utils.tensorboard.SummaryWriter, optional
        An optional SummaryWriter for logging metrics to TensorBoard, by default None.
    fold : str or int, optional
        An identifier for the current fold in cross-validation, used for
        logging purposes, by default "".

    Returns
    -------
    nn.Module
        The fine-tuned model.
    """
    lr = hparams["proxy_learning_rate"]
    weight_decay = hparams["proxy_learning_rate_decay"]
    epochs = hparams["proxy_epochs"]
    scale = hparams["scale"]
    embedding_dim = hparams["embedding_dim"]

    model.classifier = None
    proxy_loss = ProxyNCALoss(
        num_classes=num_classes, embedding_dim=embedding_dim, scale=scale
    ).to(device)
    optimizer = optim.Adam(
        list(model.parameters()) + list(proxy_loss.parameters()), lr=lr
    )
    scheduler = ExponentialLR(optimizer, gamma=weight_decay)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device).long()
            optimizer.zero_grad()
            emb = model(data, return_embedding=True, no_norm=True)
            loss = proxy_loss(emb, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
        avg_loss = total_loss / len(dataloader.dataset)
        if writer is not None:
            writer.add_scalar(f"Finetune/Loss{fold}", avg_loss, epoch)
        print(f"[Fine-tune] Epoch {epoch + 1}/{epochs}  Loss={avg_loss:.4f}")
        scheduler.step()
    return model


def create_embeddings(
    model: nn.Module, X_train: np.ndarray, X_test: np.ndarray, hparams: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function creates embedding vectors using an already trained model

    Parameters
    ----------
    model : nn.Module
        The trained model used to generate embeddings. The model's `forward`
        method must accept `return_embedding=True` and `no_norm=False` arguments.
    X_train : np.ndarray
        The training data, expected to be an array of individual samples.
        Shape should be (n_samples, ...), where ... is the sample shape.
    X_test : np.ndarray
        The test data, expected to be an array of individual samples.
        Shape should be (n_samples_test, ...).
    hparams : dict[str, any]
        A dictionary of hyperparameters. It must contain the key "embedding_dim"
        to correctly reshape the output embeddings.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two NumPy arrays:
        - The first array holds the embeddings for the training data, with a
          shape of (n_samples, embedding_dim).
        - The second array holds the embeddings for the test data, with a
          shape of (n_samples_test, embedding_dim).
    """
    embeddings = []
    test_embeddings = []

    with torch.no_grad():
        for epoch in X_train:
            embeddings.append(
                model(
                    torch.tensor(
                        epoch, dtype=torch.float, requires_grad=False
                    ).unsqueeze(0),
                    return_embedding=True,
                    no_norm=False,
                )
            )
        for epoch in X_test:
            test_embeddings.append(
                model(
                    torch.tensor(
                        epoch, dtype=torch.float, requires_grad=False
                    ).unsqueeze(0),
                    return_embedding=True,
                    no_norm=False,
                )
            )
    embd = (
        torch.stack(embeddings)
        .reshape((X_train.shape[0], hparams["embedding_dim"]))
        .numpy()
    )
    test_embd = (
        torch.stack(test_embeddings)
        .reshape((X_test.shape[0], hparams["embedding_dim"]))
        .numpy()
    )
    return embd, test_embd


def fine_tune_arcface(
    model: Union[ECAPA_TDNN, XVectorEmbeddingModel],
    hparams: Dict[str, Any],
    device: Union[torch.device, str],
    dataloader: DataLoader,
    num_classes: int,
    writer: Optional[SummaryWriter] = None,
    return_final_loss: bool = False,
    fold: Union[str, int] = "",
) -> Union[
    Union[ECAPA_TDNN, XVectorEmbeddingModel],
    Tuple[Union[ECAPA_TDNN, XVectorEmbeddingModel], float],
]:
    """
    Function training an exisiting model using ArcFace loss

    Parameters
    ----------
    model : nn.Module
        The model to be fine-tuned.
    hparams : dict[str, any]
        A dictionary of hyperparameters. Expected keys include:
        "proxy_learning_rate" (float), "proxy_learning_rate_decay" (float),
        "proxy_epochs" (int), "embedding_dim" (int), and optionally "scale" (float),
        "margin" (float), and "easy_margin" (bool).
    device : torch.device or str
        The device (e.g., "cpu" or "cuda") on which to train the model.
    dataloader : torch.utils.data.DataLoader
        The data loader for the fine-tuning data.
    num_classes : int
        The number of classes for the ArcFace loss function.
    writer : torch.utils.tensorboard.SummaryWriter, optional
        An optional SummaryWriter for logging metrics to TensorBoard, by default None.
    return_final_loss : bool, optional
        If True, the function returns the model and the final average loss as a
        tuple. Otherwise, it returns only the model, by default False.
    fold : str or int, optional
        An identifier for the current fold in cross-validation, used for
        logging purposes, by default "".

    Returns
    -------
    nn.Module or tuple[nn.Module, float]
        - If `return_final_loss` is False, returns the fine-tuned model.
        - If `return_final_loss` is True, returns a tuple containing the
          fine-tuned model and the average loss from the final epoch.
    """

    lr = hparams["proxy_learning_rate"]
    decay = hparams["proxy_learning_rate_decay"]
    epochs = hparams["proxy_epochs"]
    emb_dim = hparams["embedding_dim"]

    model.classifier = None

    arcface = ArcMarginProduct(
        in_features=emb_dim,
        out_features=num_classes,
        s=hparams.get("scale", 1.0),
        m=hparams.get("margin", 0.001),
        easy_margin=hparams.get("easy_margin", False),
    ).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(arcface.parameters()), lr=lr
    )
    scheduler = ExponentialLR(optimizer, gamma=decay)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device).long()
            optimizer.zero_grad()
            emb = model(data, return_embedding=True, no_norm=False)
            loss = arcface(emb, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        if writer:
            writer.add_scalar(f"Finetune/ArcFaceLoss{fold}", avg_loss, epoch)
        print(f"[ArcFace Fine‑tune] Epoch {epoch+1}/{epochs}  Loss={avg_loss:.4f}")
        scheduler.step()
    if return_final_loss:
        return model, avg_loss
    return model
