from sklearn.model_selection import train_test_split
from eeg_lib.commons.constant import RESULTS_FOLDER

from numpy import ndarray
from pandas.core.arrays import ExtensionArray
from sklearn.model_selection import train_test_split
from eeg_lib.commons.constant import RESULTS_FOLDER

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Optional, Tuple, Union, Any
from datetime import datetime
import os
import pandas as pd
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Optional, Tuple, Any
from datetime import datetime
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        float: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def print_train_time(start: float, end: float, device: Optional[str] = None) -> float:
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format).
        end (float): End time of computation.
        device (str, optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


def set_seeds(seed: int = 42) -> None:
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_device() -> str:
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
    return SummaryWriter(log_dir=log_dir)  # type: ignore[no-untyped-call]


def split_train_test(
    eeg_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame, Any, Any]:
    """
    Split EEG data into train and test sets based on participants.

    Args:
        eeg_df (pd.DataFrame): DataFrame containing EEG data with 'participant_id' column
        test_size (float): Proportion of data to use for testing (default 0.2)
        random_state (int): Random seed for reproducibility (default 0)

    Returns:
        Tuple containing:
            - train_df: Training set DataFrame
            - test_df: Test set DataFrame
            - train_participants: List of participant IDs in training set
            - test_participants: List of participant IDs in test set
    """
    unique_participants = eeg_df["participant_id"].unique()

    train_participants, test_participants = train_test_split(
        unique_participants, test_size=test_size, random_state=random_state
    )

    train_df = eeg_df[eeg_df["participant_id"].isin(train_participants)].reset_index(
        drop=True
    )
    test_df = eeg_df[eeg_df["participant_id"].isin(test_participants)].reset_index(
        drop=True
    )

    print("Training set participants:", train_participants)
    print("Test set participants:", test_participants)

    return train_df, test_df, train_participants, test_participants
