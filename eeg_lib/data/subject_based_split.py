from eeg_lib.data.triplet_dataset import TripletEEGDataset, HardTripletEEGDataset, SimpleEEGDataset
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
def split_by_user(X, y, labels, test_size=0.3, random_state=42):
    """
    Splits EEG data by user - training users are not in test set

    Returns:
        (X_train, y_train), (X_test, y_test)
    """
    unique_users = np.unique(y.numpy())
    train_users, test_users = train_test_split(unique_users, test_size=test_size, random_state=random_state)

    # Index selection
    train_idx = [i for i, label in enumerate(y) if label.item() in train_users]
    test_idx  = [i for i, label in enumerate(y) if label.item() in test_users]

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx], labels[train_idx], labels[test_idx]

def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32):
    """
    Wraps datasets and returns DataLoader objects.
    """
    train_dataset = TripletEEGDataset(X_train, y_train)
    test_dataset  = TripletEEGDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def create_online_dataloaders(X_train, y_train, X_test, y_test, batch_size=32):
    """
    Wraps datasets and returns DataLoader objects.
    """
    train_dataset = SimpleEEGDataset(X_train, y_train)
    test_dataset  = SimpleEEGDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
