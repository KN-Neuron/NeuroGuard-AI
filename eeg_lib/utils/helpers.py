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


def train_test_split_eeg(df, test_size=0.2, random_state=42):
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

    def __init__(self, eeg_df):
        """
        Initialize the EEG triplet dataset.

        Parameters:
        -----------
        eeg_df : pandas.DataFrame
            DataFrame containing EEG data with 'participant_id', 'epoch', and 'label' columns
        """
        self.eeg_df = eeg_df

        # Group indices by participant_id
        self.participant_to_indices = {}
        for idx, (_, row) in enumerate(eeg_df.iterrows()):
            participant_id = row['participant_id']
            if participant_id not in self.participant_to_indices:
                self.participant_to_indices[participant_id] = []
            self.participant_to_indices[participant_id].append(idx)

        # Get list of unique participant IDs
        self.participants = list(self.participant_to_indices.keys())

    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.eeg_df)

    def __getitem__(self, idx):
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

        anchor = torch.tensor(anchor_row['epoch'], dtype=torch.float32)
        positive = torch.tensor(positive_row['epoch'], dtype=torch.float32)
        negative = torch.tensor(negative_row['epoch'], dtype=torch.float32)

        return anchor, positive, negative


def save_model(model, model_name, save_dir="saved_models"):
    """
    Save a trained model to disk.

    Parameters:
    -----------
    model : torch.nn.Module
        Trained EEGNet or FBCNet model
    model_name : str
        Name to identify the model (e.g., 'eegnet_triplet')
    save_dir : str
        Directory to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save model state dictionary
    model_path = os.path.join(save_dir, f"{model_name}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Optionally save the entire model
    full_model_path = os.path.join(save_dir, f"{model_name}_full.pt")
    torch.save(model, full_model_path)
    print(f"Full model saved to {full_model_path}")


def load_model(model_class, model_path, num_channels, num_samples, embedding_size=32, device="cpu"):
    """
    Load a saved model for inference.

    Parameters:
    -----------
    model_class : class
        Model class (EEGNet or FBCNet)
    model_path : str
        Path to the saved model file
    num_channels : int
        Number of EEG channels
    num_samples : int
        Number of time samples
    embedding_size : int
        Size of the embedding vector
    device : str
        Device to load the model to

    Returns:
    --------
    torch.nn.Module
        Loaded model ready for inference
    """
    # Initialize model architecture
    if model_class.__name__ == "FBCNet":
        model = model_class(num_channels, num_samples, embedding_size, num_bands=9)
    else:
        model = model_class(num_channels, num_samples, embedding_size)

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Set model to evaluation mode
    model.eval()
    model.to(device)

    return model


