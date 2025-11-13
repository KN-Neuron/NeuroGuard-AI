import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
def encode_features_and_labels(df):
    """
    Converts epochs from the DataFrame into tensors and encodes participant IDs as labels.

    Returns:
        X (torch.FloatTensor): EEG data, shape (N, ...)
        y (torch.LongTensor): Encoded labels, shape (N,)
        label_encoder (LabelEncoder): Fitted label encoder
    """
    # Convert epochs to tensor
    X = torch.tensor(np.stack(df['epoch'].values)).float()
    # Encode participant IDs
    le = LabelEncoder()
    y = torch.tensor(le.fit_transform(df['participant_id'].values)).long()

    return X, y, le