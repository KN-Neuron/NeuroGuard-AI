import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import os
from torch.utils.data import DataLoader


def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def print_train_time(start: float, end: float, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format).
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


def set_seeds(seed: int = 42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train_test_split_eeg(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into training and testing sets.

    Args:
        df (pandas.DataFrame): DataFrame to split.
        test_size (float, optional): Proportion of data to use for testing. Defaults to 0.2.
    """
    participant_ids = df['participant_id'].unique()

    train_ids, test_ids = train_test_split(participant_ids, test_size=test_size, random_state=random_state)

    train_particiants = df[df['participant_id'].isin(train_ids)]
    test_participants = df[df['participant_id'].isin(test_ids)]
    train_set = df[df['participant_id'].isin(train_ids)]
    test_set = df[df['participant_id'].isin(test_ids)]

    return train_set, test_set, train_particiants, test_participants


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
            participant_id = row['participant_id']
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
        anchor_participant = anchor_row['participant_id']

        positive_indices = [i for i in self.participant_to_indices[anchor_participant] if i != idx]

        negative_participants = [p for p in self.participants if p != anchor_participant]
        negative_indices = []
        for p in negative_participants:
            negative_indices.extend(self.participant_to_indices[p])

        positive_idx = np.random.choice(positive_indices)
        negative_idx = np.random.choice(negative_indices)

        positive_row = self.eeg_df.iloc[positive_idx]
        negative_row = self.eeg_df.iloc[negative_idx]

        anchor = torch.tensor(anchor_row['epoch'], dtype=torch.float32).unsqueeze(0)
        positive = torch.tensor(positive_row['epoch'], dtype=torch.float32).unsqueeze(0)
        negative = torch.tensor(negative_row['epoch'], dtype=torch.float32).unsqueeze(0)

        return anchor, positive, negative


def create_user_profiles(embeddings_2d: np.ndarray, participant_ids: np.ndarray) -> dict[str:[float]]:
    """
    Create a dictionary of user profiles from 2D embeddings and participant IDs.

    Parameters
    ----------
    embeddings_2d : np.ndarray
        2D embeddings of EEG data points
    participant_ids : np.ndarray
        Array of participant IDs matching the embeddings

    Returns
    -------
    dict[str:[float]]
        Dictionary of user profiles where each key is a participant ID and the
        value is a 2D array representing the mean embedding of that participant
    """
    unique_participants = np.unique(participant_ids)
    user_profiles = {}

    for participant in unique_participants:
        mask = participant_ids == participant
        user_profiles[participant] = embeddings_2d[mask].mean(axis=0)

    return user_profiles


def predict_ids(embeddings_2d: np.ndarray, user_profiles: dict) -> list:
    """
    Predict the participant IDs of EEG data points given their embeddings and a dictionary of user profiles.

    Parameters
    ----------
    embeddings_2d : np.ndarray
        2D embeddings of EEG data points
    user_profiles : dict
        Dictionary of user profiles where each key is a participant ID and the
        value is a 2D array representing the mean embedding of that participant

    Returns
    -------
    list
        List of predicted participant IDs
    """
    predicted_ids = []

    for embedding in embeddings_2d:
        # Find the closest user profile
        closest_participant = min(user_profiles.keys(), key=lambda pid: np.linalg.norm(embedding - user_profiles[pid]))
        predicted_ids.append(closest_participant)

    return predicted_ids
