import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_eegnet(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    triplet_loss,
    num_epochs: int = 30,
) -> dict:
    """
    Train an EEGNet model using a triplet loss function.

    Args:
        model (nn.Module): EEGNet model to be trained.
        train_loader (DataLoader): DataLoader containing EEG data in the format
            (anchor, positive, negative).
        optimizer (torch.optim.Optimizer): Optimizer to be used for training.
        device (torch.device): Device to run training on.
        triplet_loss: Triplet loss function to be used.
        num_epochs (int, optional): Number of epochs to train for. Defaults to 30.

    Returns:
        dict: Dictionary containing training history (train_loss).
    """

    train_history = {"train_loss": []}

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for anchor, positive, negative in train_loader:
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)
            loss = triplet_loss(anchor_emb, positive_emb, negative_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_history["train_loss"].append(avg_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_loss:.4f}")

    return train_history
