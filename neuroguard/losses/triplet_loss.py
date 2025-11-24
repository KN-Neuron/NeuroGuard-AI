import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple


class TripletLoss(nn.Module):
    """
    Triplet loss function for learning embeddings.

    Given an anchor, positive, and negative examples, this loss function ensures that:
    - The anchor and positive examples are closer in embedding space
    - The anchor and negative examples are farther apart in embedding space
    - The margin constraint is satisfied: d(anchor, positive) + margin < d(anchor, negative)
    """

    def __init__(self, margin: float = 1.0):
        """
        Initialize the TripletLoss.

        Args:
            margin: The minimum difference between the positive and negative distances. Default: 1.0.
        """
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the triplet loss.

        Args:
            anchor: Anchor embeddings, shape (batch_size, embedding_dim)
            positive: Positive embeddings, shape (batch_size, embedding_dim)
            negative: Negative embeddings, shape (batch_size, embedding_dim)

        Returns:
            Scalar loss value
        """
        # Compute squared distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        # Compute triplet loss
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return torch.mean(loss)


class OnlineTripletLoss(nn.Module):
    """
    Online triplet loss function that mines hardest positive and negative samples within a batch.
    """

    def __init__(self, margin: float = 1.0, swap: bool = False):
        """
        Initialize the OnlineTripletLoss.

        Args:
            margin: The minimum difference between the positive and negative distances. Default: 1.0.
            swap: Whether to use distance swap. Default: False.
        """
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.swap = swap

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the online triplet loss.

        Args:
            embeddings: Embeddings, shape (batch_size, embedding_dim)
            labels: Labels, shape (batch_size,)

        Returns:
            Scalar loss value
        """
        # Compute pairwise distances
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

        # Get mask for positive pairs (same label)
        mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_neg = ~mask_pos

        # For each anchor, find hardest positive and negative
        pos_distances = pairwise_dist.masked_fill(~mask_pos, float("inf"))
        neg_distances = pairwise_dist.masked_fill(~mask_neg, float("-inf"))

        # Get hardest positive and negative (excluding self)
        mask_not_self = ~torch.eye(
            labels.size(0), dtype=torch.bool, device=labels.device
        )

        pos_distances = pos_distances.masked_fill(~mask_not_self, float("-inf"))
        neg_distances = neg_distances.masked_fill(~mask_not_self, float("-inf"))

        hardest_pos_dist = torch.max(pos_distances, dim=1)[0]
        hardest_neg_dist = torch.min(neg_distances, dim=1)[0]

        # Compute loss
        loss = torch.clamp(hardest_pos_dist - hardest_neg_dist + self.margin, min=0.0)
        return torch.mean(loss)
