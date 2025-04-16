from sklearn.model_selection import train_test_split
from eeg_lib.commons.constant import RESULTS_FOLDER

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Optional
from datetime import datetime
import os
import numpy as np
import pandas as pd


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


def get_device():
    """
    Determines the best available device for computation.

    Returns:
        str: The device to be used for computation. It returns "cuda" if a CUDA-enabled GPU is available,
             "mps" if an Apple Silicon GPU is available, and "cpu" if neither is available.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.mps.is_available():
        return "mps"
    return "cpu"


def create_writer(
    experiment_name: str, model_name: str, extra: Optional[str] = None
) -> SummaryWriter:
    """Creates a SummaryWriter instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        SummaryWriter: Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "eeg_lib/data/result_summaries/2025-03-15/data_10_percent/EEGNet/50_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="EEGNet",
                               extra="50_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="eeg_lib/data/result_summaries/runs/2025-03-15/data_10_percent/EEGNet/50_epochs/")
    """

    timestamp = datetime.now().strftime("%Y-%m-%d")

    if extra:
        log_dir = os.path.join(
            f"{RESULTS_FOLDER}/runs", timestamp, experiment_name, model_name, extra
        )
    else:
        log_dir = os.path.join(
            f"{RESULTS_FOLDER}/runs", timestamp, experiment_name, model_name
        )

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)


def split_train_test(
    eeg_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits an EEG DataFrame into training and testing sets based on unique participant IDs.

    This function identifies all unique participant IDs in the given DataFrame, then splits these
    unique IDs into training and test groups using the specified test size and random state. The
    function then filters the DataFrame to obtain training and testing subsets, ensuring that participants in those subsets are separate.The EEG epoch
    data and corresponding participant IDs are returned as numpy arrays.

    Args:
        eeg_df (pd.DataFrame): DataFrame containing EEG data. It must include at least:
            - "participant_id": Identifier for each participant (e.g., string or integer).
            - "epoch": EEG epoch data (e.g., a list or numpy array of shape (channels, timesteps)).
        test_size (float, optional): Proportion of unique participants to include in the test set.
            Defaults to 0.2.
        random_state (int, optional): Random seed for reproducible splitting of participant IDs.
            Defaults to 0.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - X_train: Numpy array of EEG epochs for training.
            - X_test: Numpy array of EEG epochs for testing.
            - y_train: Numpy array of participant IDs corresponding to training epochs.
            - y_test: Numpy array of participant IDs corresponding to test epochs.

    Example:
        >>> X_train, X_test, y_train, y_test = split_train_test(eeg_df, test_size=0.2, random_state=42)
        >>> print("Train participants:", np.unique(y_train))
        >>> print("Test participants:", np.unique(y_test))
    """

    unique_participants = np.unique(eeg_df["participant_id"])

    train_participants, test_participants = train_test_split(
        unique_participants, test_size=test_size, random_state=random_state
    )

    train_df = eeg_df[eeg_df["participant_id"].isin(train_participants)].reset_index(drop=True)
    test_df = eeg_df[eeg_df["participant_id"].isin(test_participants)].reset_index(drop=True)

    y_train = train_df["participant_id"].values

    y_test = test_df["participant_id"].values
    X_train = train_df["epoch"].values
    X_test = test_df["epoch"].values

    print("Training set participants:", train_participants)
    print("Test set participants:", test_participants)
    print("Training labels:", np.unique(y_train))
    print("Test labels:", np.unique(y_test))

    return X_train, X_test, y_train, y_test