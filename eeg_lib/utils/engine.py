import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from eeg_lib.models.similarity.eegnet import EEGNetEmbeddingModel


def train_eegnet(
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        triplet_loss,
        num_epochs: int = 30,
) -> dict[str, list]:
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


def generate_embeddings_2d(eegnet_model: nn.Module, test_loader: DataLoader, device: torch.device) -> np.ndarray:
    """
    Generate test embeddings using a trained EEGNet model.

    Args:
        eegnet_model (nn.Module): Trained EEGNet model.
        test_loader (DataLoader): DataLoader containing EEG data in the format
            (anchor, positive, negative).
        device (torch.device): Device to run inference on.

    Returns:
        np.ndarray: 2D array with shape (num_test_samples, embedding_dim) containing
            the embeddings for all test samples.
    """
    eegnet_model.eval()
    test_embeddings = []

    with torch.no_grad():
        for anchor, positive, negative in test_loader:
            anchor = anchor.to(device)
            embeddings = eegnet_model(anchor)
            test_embeddings.append(embeddings.cpu().numpy())

    test_embeddings = np.concatenate(test_embeddings, axis=0)
    return test_embeddings
