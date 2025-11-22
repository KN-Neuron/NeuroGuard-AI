import torch.nn as nn
import math
import torch
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    """
    Implements the ArcFace loss (Additive Angular Margin Loss).

    This module enhances class separability by adding an angular margin
    to the logits before applying the softmax cross-entropy loss.

    Parameters
    ----------
    in_features : int
        Dimensionality of the input embeddings.
    out_features : int
        Number of output classes.
    s : float, optional, default=30.0
        Scale factor applied to logits before softmax.
    m : float, optional, default=0.50
        Additive angular margin in radians.
    easy_margin : bool, optional, default=False
        If True, applies the easy margin strategy to avoid numerical instability.

    Forward
    -------
    embeddings : torch.Tensor
        Tensor of shape (batch_size, in_features)
        Input feature vectors to be classified.
    labels : torch.LongTensor
        Tensor of shape (batch_size,)
        Ground-truth class indices for each embedding.

    Returns
    -------
    torch.Tensor
        Scalar tensor representing the cross-entropy loss
        after applying ArcFace margin transformation.
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.easy_margin = easy_margin

    def forward(
        self, embeddings: torch.Tensor, labels: torch.LongTensor
    ) -> torch.Tensor:
        """
        Forward pass through the ArcMarginProduct layer.

        Parameters
        ----------
        embeddings : torch.Tensor
            Input embedding vectors of shape (batch_size, in_features).
        labels : torch.LongTensor
            class indices for each embedding.

        Returns
        -------
        torch.Tensor
            Cross-entropy loss computed with ArcFace angular margins.
        """
        normalized_emb = F.normalize(embeddings, p=2, dim=1)
        normalized_w = F.normalize(self.weight, p=2, dim=1)

        cosine = F.linear(normalized_emb, normalized_w)
        radicand = torch.clamp(1.0 - cosine**2, min=0.0)
        sine = torch.sqrt(radicand)

        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s

        loss = F.cross_entropy(logits, labels)
        return loss
