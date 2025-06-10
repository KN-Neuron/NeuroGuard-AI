import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, num_classes, embedding_dim, scale=None):
        super(ProxyNCALoss, self).__init__()
        # Create a learnable embedding table of shape (num_classes, embedding_dim)
        self.proxies = nn.Embedding(num_classes, embedding_dim)
        # Initialize proxy weights with Kaiming normal (fan-out)
        nn.init.kaiming_normal_(self.proxies.weight, mode='fan_out')

        # Set or create the scaling factor for similarities
        if scale is not None:
            # If user provided a fixed scale (float), use it directly
            self.scale = scale
        else:
            # Otherwise, make it a learnable parameter initialized to 1.0
            self.scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

    def forward(self, embeddings, labels):
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
        # 1) Normalize each sample embedding to unit length
        embeddings = F.normalize(embeddings, p=2, dim=1)  # (B, D)

        # 2) Normalize each proxy vector to unit length
        proxies = F.normalize(self.proxies.weight, p=2, dim=1)  # (C, D)

        # 3) Compute cosine similarities between each embedding and each proxy
        #    Resulting shape: (batch_size, num_classes)
        similarities = torch.matmul(embeddings, proxies.t())

        # 4) Scale the similarities (to control the "softness" of the softmax)
        logits = self.scale * similarities

        # 5) Compute standard cross-entropy loss over the scaled logits
        loss = nn.CrossEntropyLoss()(logits, labels)

        return loss
