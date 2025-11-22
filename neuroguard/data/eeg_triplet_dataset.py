import torch
import pandas as pd
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class EEGTripletDataset(Dataset):
    """
    Dataset for generating triplets (anchor, positive, negative) for EEG data.

    This dataset takes a pandas DataFrame containing EEG data with participant_id,
    epoch (EEG signal), and label columns, and generates triplets for training
    with triplet margin loss.
    """

    def __init__(self, eeg_df: pd.DataFrame):
        """
        Initialize the EEG triplet dataset.

        Parameters:
        -----------
        eeg_df : pandas.DataFrame
            DataFrame containing EEG data with 'participant_id', 'epoch', and 'label' columns
        """
        self.eeg_df = eeg_df

        self.participant_to_indices = {}
        for idx, (_, row) in enumerate(eeg_df.iterrows()):
            participant_id = row["participant_id"]
            if participant_id not in self.participant_to_indices:
                self.participant_to_indices[participant_id] = []
            self.participant_to_indices[participant_id].append(idx)

        self.participants = list(self.participant_to_indices.keys())

    def __len__(self) -> int:
        """Return the total number of samples in the dataset"""
        return len(self.eeg_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a triplet (anchor, positive, negative) for training.

        Returns:
        --------
        tuple
            Tuple containing (anchor, positive, negative) tensors
        """
        anchor_row = self.eeg_df.iloc[idx]
        anchor_participant = anchor_row["participant_id"]

        positive_indices = [
            i for i in self.participant_to_indices[anchor_participant] if i != idx
        ]

        negative_participants = [
            p for p in self.participants if p != anchor_participant
        ]
        negative_indices = []
        for p in negative_participants:
            negative_indices.extend(self.participant_to_indices[p])

        positive_idx = np.random.choice(positive_indices)
        negative_idx = np.random.choice(negative_indices)

        positive_row = self.eeg_df.iloc[positive_idx]
        negative_row = self.eeg_df.iloc[negative_idx]

        anchor = torch.tensor(anchor_row["epoch"], dtype=torch.float32).unsqueeze(0)
        positive = torch.tensor(positive_row["epoch"], dtype=torch.float32).unsqueeze(0)
        negative = torch.tensor(negative_row["epoch"], dtype=torch.float32).unsqueeze(0)

        return anchor, positive, negative
