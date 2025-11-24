"""EEG subject-based data splitting utilities."""

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import Any, Tuple
from ..types import EEGDataTensor
from .triplet_dataset import TripletEEGDataset, SimpleEEGDataset


def split_by_user(
    X: EEGDataTensor,
    y: torch.Tensor,
    labels: torch.Tensor,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[
    EEGDataTensor, torch.Tensor, EEGDataTensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Splits EEG data by user - training users are not in test set

    Args:
        X: EEG data tensor of shape (n_samples, n_channels, n_time_points)
        y: User labels tensor of shape (n_samples,)
        labels: Additional labels tensor of shape (n_samples,)
        test_size: Proportion of users to use for testing (default 0.3)
        random_state: Random seed for reproducibility (default 42)

    Returns:
        Tuple containing:
            - X_train: Training EEG data
            - y_train: Training user labels
            - X_test: Test EEG data
            - y_test: Test user labels
            - labels_train: Training additional labels
            - labels_test: Test additional labels
    """
    unique_users = np.unique(y.numpy())
    train_users, test_users = train_test_split(
        unique_users, test_size=test_size, random_state=random_state
    )

    train_idx = [i for i, label in enumerate(y) if label.item() in train_users]
    test_idx = [i for i, label in enumerate(y) if label.item() in test_users]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    labels_train = labels[train_idx]
    labels_test = labels[test_idx]

    return X_train, y_train, X_test, y_test, labels_train, labels_test


def create_dataloaders(
    X_train: EEGDataTensor,
    y_train: torch.Tensor,
    X_test: EEGDataTensor,
    y_test: torch.Tensor,
    batch_size: int = 32,
) -> Tuple[DataLoader[Any], DataLoader[Any]]:
    """
    Create data loaders for triplet training.

    Args:
        X_train: Training EEG data
        y_train: Training labels
        X_test: Test EEG data
        y_test: Test labels
        batch_size: Batch size for data loading (default 32)

    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_dataset = TripletEEGDataset(X_train, y_train)
    test_dataset = TripletEEGDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def create_online_dataloaders(
    X_train: EEGDataTensor,
    y_train: torch.Tensor,
    X_test: EEGDataTensor,
    y_test: torch.Tensor,
    batch_size: int = 32,
) -> Tuple[DataLoader[Any], DataLoader[Any]]:
    """
    Create data loaders for online training.

    Args:
        X_train: Training EEG data
        y_train: Training labels
        X_test: Test EEG data
        y_test: Test labels
        batch_size: Batch size for data loading (default 32)

    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_dataset = SimpleEEGDataset(X_train, y_train)
    test_dataset = SimpleEEGDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
