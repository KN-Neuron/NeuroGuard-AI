import os
import random
from typing import Tuple
import pandas as pd

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from eeg_lib.commons.constant import NUM_WORKERS


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
):
    """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """

    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    class_names = train_data.classes

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names


class EEGNetColorDataset(Dataset):
    """
    A simple Dataset for EEG data stored in a DataFrame.

    Each row of the DataFrame has:
      - 'epoch': a NumPy array of shape (4, 751).
      - 'label': a string or numeric label (e.g., color name).
    """

    def __init__(self, df, label_to_idx=None, transform=None):
        """
        Args:
            df (pandas.DataFrame): Must have 'epoch' and 'label' columns.
            label_to_idx (dict): Optional map from label string to int.
                                 If None, one is created automatically.
            transform (callable): Optional transform function for data augmentation or normalization.
        """
        self.df: pd.DataFrame = df.reset_index(drop=True)
        self.transform = transform

        if label_to_idx is not None:
            self.label_to_idx = label_to_idx
        else:
            unique_labels = sorted(self.df["label"].unique())
            self.label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        # epoch_data is shape (4, 751)
        epoch_data = row["epoch"]

        x = torch.tensor(epoch_data, dtype=torch.float32)
        # Reshape to (1, 4, 751) to match EEGNet's expected input shape
        x = x.unsqueeze(0)

        if self.transform:
            x = self.transform(x)

        label_str = row["label"]
        y = self.label_to_idx[label_str]
        y = torch.tensor(y, dtype=torch.long)  # if using standard classification

        return x, y


class TripletEEGDataset(Dataset):
    """
    A Dataset that returns (anchor, positive, negative) EEG epochs for triplet training.

    We assume:
      - 'df' is a DataFrame with columns: ['participant_id', 'epoch'].
      - 'epoch' is shape (4, 751).
      - We want anchor and positive from the same participant,
        negative from a different participant.
    """

    def __init__(self, df):
        """
        Args:
            df (pd.DataFrame): Must contain 'participant_id' and 'epoch' columns.
        """
        self.df = df.reset_index(drop=True)

        self.participants_dict = {}
        for idx, row in self.df.iterrows():
            pid = row["participant_id"]
            if pid not in self.participants_dict:
                self.participants_dict[pid] = []
            self.participants_dict[pid].append(idx)

        self.participant_ids = sorted(list(self.participants_dict.keys()))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        Returns a tuple: (anchor_tensor, positive_tensor, negative_tensor).
        Each is shape (1, 4, 751) if we add the channel dimension.
        """
        anchor_row = self.df.iloc[index]
        anchor_pid = anchor_row["participant_id"]

        anchor_tensor = torch.tensor(
            anchor_row["epoch"], dtype=torch.float32
        ).unsqueeze(0)

        pos_index = index
        while pos_index == index:
            pos_index = random.choice(self.participants_dict[anchor_pid])
        pos_row = self.df.iloc[pos_index]
        positive_tensor = torch.tensor(pos_row["epoch"], dtype=torch.float32).unsqueeze(
            0
        )

        neg_pid = anchor_pid
        while neg_pid == anchor_pid:
            neg_pid = random.choice(self.participant_ids)
        neg_index = random.choice(self.participants_dict[neg_pid])
        neg_row = self.df.iloc[neg_index]
        negative_tensor = torch.tensor(neg_row["epoch"], dtype=torch.float32).unsqueeze(
            0
        )

        return anchor_tensor, positive_tensor, negative_tensor
