import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Any, Dict, Tuple


def add_gaussian_noise(eeg: np.ndarray, std: float = 0.01) -> np.ndarray:
    """
    Add Gaussian noise to an EEG sample.

    Parameters
    ----------
    eeg : np.ndarray
        Input EEG signal, shape (n_channels, n_times) or (feature_dim,).
    std : float, optional
        Standard deviation of the Gaussian noise. Default is 0.01.

    Returns
    -------
    np.ndarray
        EEG signal with added Gaussian noise.
    """
    noise = np.random.normal(0, std, eeg.shape)
    return eeg + noise


def get_dataset(
    hparams: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray
) -> DataLoader:
    """
    Create a DataLoader for training the TDNN model.

    Parameters
    ----------
    hparams : dict
        Dictionary containing hyperparameters:
            - "augmentation" : bool
                Whether to apply Gaussian noise augmentation.
            - "batch_size" : int
                Number of samples per batch.
            - "std" : float
                Standard deviation for Gaussian noise.
    X_train : np.ndarray
        Training data of shape (n_samples, n_features, ...) or (n_samples, n_channels, n_times).
    y_train : np.ndarray
        Corresponding labels of shape (n_samples,).

    Returns
    -------
    torch.utils.data.DataLoader
        DataLoader that yields mini-batches of (tensor, label) pairs.
    """
    augmentation = hparams["augmentation"]
    batch_size = hparams["batch_size"]
    std = hparams["std"]

    dataset = TDNNDataset(X_train, y_train, augmentation=augmentation, std=std)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


class TDNNDataset(Dataset):
    """
    Dataset wrapper for EEG data used in TDNN training.

    Attributes
    ----------
    data : np.ndarray
        Input EEG data of shape (n_samples, n_features, ...) or (n_samples, n_channels, n_times).
    labels : np.ndarray
        Corresponding labels for each EEG sample.
    augmentation : bool
        Whether to apply Gaussian noise augmentation.
    std : float
        Standard deviation for Gaussian noise.
    """
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        augmentation: bool = False,
        std: float = 0.01
    ) -> None:
        """
        Initialize the dataset.

        Parameters
        ----------
        data : np.ndarray
            EEG or feature data, shape (n_samples, n_features, ...).
        labels : np.ndarray
            Class labels corresponding to each EEG sample.
        augmentation : bool, optional
            If True, Gaussian noise will be added during training. Default is False.
        std : float, optional
            Standard deviation of Gaussian noise. Default is 0.01.
        """

        self.data = data
        self.labels = labels
        self.augmentation = augmentation
        self.std = std

    def __len__(self) -> int:
        """
        Get the total number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a single EEG sample and its label.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple of torch.Tensor
            A tuple (x, y) where:
                - x : torch.Tensor
                    EEG sample tensor of shape matching input features.
                - y : torch.Tensor
                    Corresponding label tensor (long dtype).
        """
        eeg = self.data[index]
        if self.augmentation:
            eeg = add_gaussian_noise(eeg, self.std)
        x = torch.tensor(eeg, dtype=torch.float32)
        y = torch.tensor(self.labels[index], dtype=torch.long)
        return x, y