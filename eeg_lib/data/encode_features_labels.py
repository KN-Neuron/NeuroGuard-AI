import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Tuple


def encode_features_and_labels(
    df: pd.DataFrame,
) -> Tuple[torch.Tensor, torch.Tensor, LabelEncoder]:
    """
    Converts epochs from the DataFrame into tensors and encodes participant IDs as labels.

    Args:
        df: DataFrame with columns 'epoch' and 'participant_id'

    Returns:
        A tuple containing:
            - X: EEG data tensor
            - y: Encoded labels tensor
            - label_encoder: Fitted label encoder
    """
    # Convert epochs to tensor - stack the epochs column which contains numpy arrays
    X = torch.tensor(
        np.stack(df["epoch"].values)
    ).float()  # Shape: (N, n_channels, n_time_points)
    # Encode participant IDs
    le = LabelEncoder()
    y = torch.tensor(le.fit_transform(df["participant_id"].values)).long()

    return X, y, le
