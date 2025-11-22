import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ProxyNCALoss(nn.Module):
    """
    Proxy Neighborhood Component Analysis (ProxyNCA) loss.

    This loss learns a set of "proxies"—one proxy vector per class—so that
    each sample embedding is encouraged to be close to its class proxy and
    far from other class proxies in the embedding space.

    Args:
        num_classes (int): Number of classes (and thus number of proxies).
        embedding_dim (int): Dimensionality of the embedding space.
        scale (float or nn.Parameter, optional): Scaling factor applied to
            cosine similarities before the softmax. If None, a learnable
            scalar parameter is initialized to 1.0.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        scale: float | nn.Parameter | None = None,
    ) -> None:
        super(ProxyNCALoss, self).__init__()
        self.proxies = nn.Embedding(num_classes, embedding_dim)
        nn.init.kaiming_normal_(self.proxies.weight, mode="fan_out")

        if scale is not None:
            self.scale = scale
        else:
            self.scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

    def forward(self, embeddings: Tensor, labels: torch.LongTensor) -> Tensor:
        """
        Compute the ProxyNCA loss.

        Args:
            embeddings (Tensor):
                Shape (batch_size, embedding_dim).  These are the model's
                output embeddings for a batch of samples.
            labels (LongTensor):
                Shape (batch_size,).  Ground-truth class indices for each
                sample, values in [0, num_classes).

        Returns:
            loss (Tensor): Scalar loss value.
        """
        embeddings = F.normalize(embeddings, p=2, dim=1)  # (B, D)

        proxies = F.normalize(self.proxies.weight, p=2, dim=1)  # (C, D)

        similarities = torch.matmul(embeddings, proxies.t())

        logits = self.scale * similarities

        loss = nn.CrossEntropyLoss()(logits, labels)

        return loss
