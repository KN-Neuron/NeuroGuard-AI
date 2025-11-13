import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Union


class TripletDataset(Dataset):
    """
    Dataset for generating triplets (anchor, positive, negative) used in triplet-loss training.

    Each sample corresponds to one EEG epoch from a participant. For a given anchor sample,
    a positive example is chosen from the same participant, and a negative example is chosen
    from a different participant.

    Attributes
    ----------
    participant_ids : np.ndarray
        1D array of participant IDs (strings or integers).
    eeg_data : np.ndarray
        Array of EEG epochs, each of shape (n_channels, n_times) or similar.
    user_id_dict : dict
        Mapping of participant ID -> list of indices corresponding to that participant's samples.
    """
    def __init__(
        self,
        participant_ids: np.ndarray,
        epochs: np.ndarray
    ) -> None:
        """
        Initialize the TripletDataset.

        Parameters
        ----------
        participant_ids : np.ndarray
            1D array of participant identifiers (e.g., subject IDs).
        epochs : np.ndarray
            EEG epochs corresponding to each participant ID.
            Shape: (n_samples, n_channels, n_times) or similar.
        """
        self.participant_ids = participant_ids
        self.eeg_data = epochs

        unique_ids = np.unique(self.participant_ids)
        self.user_id_dict = {}
        for part_id in unique_ids:
            self.user_id_dict[part_id] = np.where(self.participant_ids == part_id)[0].tolist()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve a triplet of EEG samples: (anchor, positive, negative).

        Parameters
        ----------
        index : int
            Index of the anchor sample.

        Returns
        -------
        tuple of torch.Tensor
            A tuple (anchor, positive, negative), each being a float32 tensor
            with an added channel dimension (unsqueezed at dim=0).
        """
        anchor = torch.tensor(self.eeg_data[index], dtype=torch.float).unsqueeze(0)
        anchor_label = self.participant_ids[index]

        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.user_id_dict[anchor_label])
        positive = torch.tensor(self.eeg_data[positive_index], dtype=torch.float).unsqueeze(0)

        negative_label = np.random.choice([l for l in self.user_id_dict.keys() if l != anchor_label])
        negative_index = np.random.choice(self.user_id_dict[negative_label])
        negative = torch.tensor(self.eeg_data[negative_index], dtype=torch.float).unsqueeze(0)

        return anchor, positive, negative

    def __len__(self) -> int:
        """
        Get the total number of available samples (anchors).

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return len(self.participant_ids)
