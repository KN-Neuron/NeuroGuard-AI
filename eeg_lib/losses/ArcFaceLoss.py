import torch.nn as nn
import math
import torch
import torch.nn.functional as F



class ArcMarginProduct(nn.Module):
    """
    Implements the ArcFace loss, which enhances the discriminative power of the model by adding an additive angular margin penalty between classes.

    This module normalizes the input embeddings and class weights, computes the cosine similarity,
    applies an angular margin to the target logits, scales the logits, and returns the cross-entropy loss.

    :param in_features : int - input features
    :param out_features : int - number of classes
    :param s : float -  optional (default=30.0) Scale factor applied to the logits
    :param m : float -  optional (default=0.50) - Additive angular margin (in radians) to enhance class separability.
    :param easy_margin : bool -  optional (default=False) if True uses easy margin strategy

    Forward
    :param embeddings : torch.Tensor -  shape (batch_size, in_features) Input feature vectors to be classified.
    :param labels : torch.LongTensor -  shape (batch_size,) Ground-truth class indices for each embedding.

    Returns
    loss : torch.Tensor - Scalar tensor representing the cross-entropy loss after applying ArcFace margins.
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        """
        :param in_features: input features
        :param out_features: output features
        :param s: scale factor
        :param m: margin
        :param easy_margin: whether to use easy margin strategy
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s      # feature scale
        self.m = m      # angular margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

        self.easy_margin = easy_margin

    def forward(self, embeddings, labels):
        normalized_emb = F.normalize(embeddings, p=2, dim=1)        # (B, D)
        normalized_w   = F.normalize(self.weight,   p=2, dim=1)     # (C, D)

        cosine = F.linear(normalized_emb, normalized_w)            # (B, C)
        radicand = torch.clamp(1.0 - cosine ** 2, min=0.0)
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