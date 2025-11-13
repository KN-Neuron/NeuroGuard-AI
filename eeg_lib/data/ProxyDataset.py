import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Any

class ProxyDataset(Dataset):
    """
    A simple dataset wrapper for EEG or feature-based data.

    Attributes
    ----------
    data : np.ndarray
        Input data of shape (n_samples, ...).
    labels : np.ndarray
        Corresponding labels for each sample.
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray) -> None:
        """
        Initialize the dataset.

        Parameters
        ----------
        data : np.ndarray
            Array containing the samples.
        labels : np.ndarray
            Array of corresponding labels.
        """

        self.data = data
        self.labels = labels
        print(len(np.unique(labels)))

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        """
        Retrieve a single sample and its label.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve.

        Returns
        -------
        Tuple[torch.Tensor, Any]
            Tuple containing:
              - data tensor with shape (1, feature_dim)
              - corresponding label
        """
        return torch.Tensor(self.data[index]).unsqueeze(0), self.labels[index]