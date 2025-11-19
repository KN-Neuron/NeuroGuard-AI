import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from typing import Callable, Union
from dataclasses import dataclass
from torch.utils.data import DataLoader
from typing import Any, Callable, Union

@dataclass
class TrainingConfig:
    batch_size: int = 32
    margin: float = 0.5
    epochs: int = 10
    learning_rate: float = 0.001


def train_triplet_epoch(
    model: nn.Module,
    dataloader: DataLoader[Any],
    loss_fn: Callable[..., torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        anchor, positive, negative = (x.to(device) for x in batch[:3])

        emb_anchor = model(anchor)
        emb_positive = model(positive)
        emb_negative = model(negative)

        loss = loss_fn(emb_anchor, emb_positive, emb_negative)

        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def train_triplet(model: nn.Module, train_loader: torch.utils.data.dataloader, criterion: Callable, optimizer: torch.optim, device: torch.device, n_epochs:int=10):
    for epoch in range(1, n_epochs + 1):
        loss = train_triplet_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch}/{n_epochs} | Triplet Loss: {loss:.4f}")


def pairwise_distances(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Euclidean distances between embeddings
    """
    dot_product = torch.matmul(embeddings, embeddings.t())
    square_norm = torch.diagonal(dot_product)
    distances = square_norm.unsqueeze(1) - 2 * dot_product + square_norm.unsqueeze(0)
    distances = torch.clamp(distances, min=0.0)
    distances = torch.sqrt(distances + 1e-8)
    return distances


def train_triplet_epoch_online(
    model: nn.Module,
    dataloader: DataLoader[Any],
    loss_fn: Callable[..., torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Trains a model for one epoch using triplet loss with online triplet mining.

    For each batch, dynamically selects the hardest positive and hardest negative
    samples for each anchor, computes triplet loss, and updates model parameters.

    Args:
        model (torch.nn.Module): The neural network model to train.
        dataloader (torch.utils.data.DataLoader): Dataloader providing (inputs, labels) batches.
        loss_fn (callable): Triplet loss function (e.g., `torch.nn.TripletMarginLoss`).
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        device (torch.device): Device to run computations on (e.g., "cuda" or "cpu").

    Returns:
        float: The average triplet loss for the epoch.
    """

    model.train()
    total_loss = 0.0

    for batch in dataloader:
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        embeddings = model(inputs)

        distances = pairwise_distances(embeddings)

        batch_size = labels.size(0)

        triplet_loss: Union[float, torch.Tensor] = 0.0
        triplets_count = 0

        for i in range(batch_size):
            anchor_label = labels[i]
            anchor_dist = distances[i]

            positive_mask = (labels == anchor_label) & (
                torch.arange(batch_size).to(device) != i
            )
            negative_mask = labels != anchor_label

            if positive_mask.sum() == 0 or negative_mask.sum() == 0:
                continue

            hardest_positive_dist, hardest_positive_idx = torch.max(
                anchor_dist * positive_mask.float(), dim=0
            )

            neg_distances = anchor_dist.clone()
            neg_distances[~negative_mask] = 1e6

            hardest_negative_dist, hardest_negative_idx = torch.min(
                neg_distances, dim=0
            )

            anchor_emb = embeddings[i]
            positive_emb = embeddings[hardest_positive_idx]
            negative_emb = embeddings[hardest_negative_idx]

            loss = loss_fn(
                anchor_emb.unsqueeze(0),
                positive_emb.unsqueeze(0),
                negative_emb.unsqueeze(0),
            )
            triplet_loss = (
                triplet_loss + loss if isinstance(triplet_loss, torch.Tensor) else loss
            )
            triplets_count += 1

        if triplets_count > 0 and isinstance(triplet_loss, torch.Tensor):
            triplet_loss = triplet_loss / triplets_count
            triplet_loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            total_loss += triplet_loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def train_triplet_online(model: nn.Module, train_loader: torch.utils.data.dataloader, criterion: Callable, optimizer: torch.optim, device: torch.device, n_epochs: int =10)->float:
    """
    Trains a model using train_triplet_epoch_online function
    and prints training status after every epoch

    Args:
    model (torch.nn.Module): The neural network model to train.
    train_loader (torch.utils.data.DataLoader): Dataloader for training set
    criterion: (callable): Loss function
    optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
    device (torch.device): Device to run computations on (e.g., "cuda" or "cpu").
    n_epochs (integer): Number of epochs for training

    """
    for epoch in range(1, n_epochs + 1):
        loss = train_triplet_epoch_online(
            model, train_loader, criterion, optimizer, device
        )
        print(f"Epoch {epoch}/{n_epochs} | Triplet Loss: {loss:.4f}")
    return loss

