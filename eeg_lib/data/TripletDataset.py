import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TripletDataset(Dataset):
    def __init__(self, participant_ids: np.ndarray, epochs: np.ndarray):
        """
        Args:
            participant_ids (np.ndarray): 1D array of participant IDs (strings or integers).
            epochs (np.ndarray): Array of EEG epochs. Each element should be a structure (e.g. list or np.array)
                                 of shape (4, 751), corresponding to 4 channels and 751 timesteps.
        """
        self.participant_ids = participant_ids  # e.g. np.array([...])
        self.eeg_data = epochs  # e.g. np.array([...]) with each element shaped (4, 751)

        # Build dictionary linking participant id to all indices where they occur.
        unique_ids = np.unique(self.participant_ids)
        self.user_id_dict = {}
        for part_id in unique_ids:
            # np.where returns a tuple; take the first element and convert to list.
            self.user_id_dict[part_id] = np.where(self.participant_ids == part_id)[0].tolist()

    def __getitem__(self, index):
        # Get anchor sample and its participant ID
        anchor = torch.tensor(self.eeg_data[index], dtype=torch.float).unsqueeze(0)
        anchor_label = self.participant_ids[index]

        # Select a positive sample: choose a different index with the same participant ID.
        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.user_id_dict[anchor_label])
        positive = torch.tensor(self.eeg_data[positive_index], dtype=torch.float).unsqueeze(0)

        # Select a negative sample: choose an index with a different participant ID.
        negative_label = np.random.choice([l for l in self.user_id_dict.keys() if l != anchor_label])
        negative_index = np.random.choice(self.user_id_dict[negative_label])
        negative = torch.tensor(self.eeg_data[negative_index], dtype=torch.float).unsqueeze(0)

        return anchor, positive, negative

    def __len__(self):
        return len(self.participant_ids)
