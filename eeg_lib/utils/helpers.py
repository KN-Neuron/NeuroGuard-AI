import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import datasets, transforms
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


def prepare_eeg_data(eeg_df, test_size=0.2, val_size=0.1, random_state=42):
    """
    Prepare EEG data from dataframe for EEGNet and FBCNet models.

    Parameters:
    -----------
    eeg_df : pandas.DataFrame
        DataFrame containing EEG data with 'participant_id', 'epoch', and 'label' columns
    test_size : float, default=0.2
        Proportion of participants to include in test set
    val_size : float, default=0.1
        Proportion of participants to include in validation set
    random_state : int, default=42
        Random seed for reproducibility

    Returns:
    --------
    dict
        Dictionary containing train, validation, and test data and labels,
        along with participant mapping information
    """

    participant_ids = eeg_df['participant_id'].unique()

    train_val_ids, test_ids = train_test_split(
        participant_ids,
        test_size=test_size,
        random_state=random_state
    )

    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_size / (1 - test_size),
        random_state=random_state
    )

    print(f"Train participants: {len(train_ids)}")
    print(f"Validation participants: {len(val_ids)}")
    print(f"Test participants: {len(test_ids)}")

    train_data = eeg_df[eeg_df['participant_id'].isin(train_ids)]
    val_data = eeg_df[eeg_df['participant_id'].isin(val_ids)]
    test_data = eeg_df[eeg_df['participant_id'].isin(test_ids)]

    X_train, y_train, user_train = process_eeg_set(train_data)
    X_val, y_val, user_val = process_eeg_set(val_data)
    X_test, y_test, user_test = process_eeg_set(test_data)

    label_mapping = {label: idx for idx, label in enumerate(sorted(eeg_df['label'].unique()))}
    user_mapping = {user_id: idx for idx, user_id in enumerate(sorted(participant_ids))}

    y_train_idx = np.array([label_mapping[label] for label in y_train])
    y_val_idx = np.array([label_mapping[label] for label in y_val])
    y_test_idx = np.array([label_mapping[label] for label in y_test])

    user_train_idx = np.array([user_mapping[user] for user in user_train])
    user_val_idx = np.array([user_mapping[user] for user in user_val])
    user_test_idx = np.array([user_mapping[user] for user in user_test])

    return {
        'X_train': X_train,
        'y_train': y_train_idx,
        'user_train': user_train_idx,
        'X_val': X_val,
        'y_val': y_val_idx,
        'user_val': user_val_idx,
        'X_test': X_test,
        'y_test': y_test_idx,
        'user_test': user_test_idx,
        'label_mapping': label_mapping,
        'user_mapping': user_mapping,
        'inv_label_mapping': {v: k for k, v in label_mapping.items()},
        'inv_user_mapping': {v: k for k, v in user_mapping.items()}
    }


def process_eeg_set(data_df):
    """
    Process a subset of EEG data into numpy arrays.

    Parameters:
    -----------
    data_df : pandas.DataFrame
        DataFrame containing EEG data

    Returns:
    --------
    tuple
        (X, y, user_ids) where X is the EEG data, y is the labels, and user_ids are participant IDs
    """
    X = []
    y = []
    user_ids = []

    for _, row in data_df.iterrows():
        # Extract EEG signal from the 'epoch' column
        eeg_signal = np.array(row['epoch'])

        # Ensure the signal is properly shaped (channels, samples)
        if len(eeg_signal.shape) == 1:
            # If it's a 1D array, reshape assuming it's a single channel
            eeg_signal = eeg_signal.reshape(1, -1)

        X.append(eeg_signal)
        y.append(row['label'])
        user_ids.append(row['participant_id'])

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    user_ids = np.array(user_ids)

    return X, y, user_ids


def apply_bandpass_filters(X, fs=256, filter_bands=None):
    """
    Apply bandpass filters to EEG data for FBCNet preprocessing.

    Parameters:
    -----------
    X : numpy.ndarray
        EEG data with shape (trials, channels, samples)
    fs : int, default=256
        Sampling frequency in Hz
    filter_bands : list of tuples, default=None
        List of frequency bands as (low_freq, high_freq) tuples
        If None, uses standard EEG bands

    Returns:
    --------
    numpy.ndarray
        Filtered EEG data with shape (trials, bands, channels, samples)
    """
    if filter_bands is None:
        # Default 9 frequency bands as in the file
        filter_bands = [
            (0.5, 4),  # Delta
            (4, 8),  # Theta
            (8, 10),  # Alpha1
            (10, 12),  # Alpha2
            (12, 16),  # Beta1
            (16, 20),  # Beta2
            (20, 30),  # Beta3
            (30, 40),  # Gamma1
            (40, 100)  # Gamma2
        ]

    num_trials, num_channels, num_samples = X.shape
    filtered_data = np.zeros((num_trials, len(filter_bands), num_channels, num_samples))

    for band_idx, (low_freq, high_freq) in enumerate(filter_bands):
        # Design Butterworth bandpass filter
        nyquist = 0.5 * fs
        low = low_freq / nyquist
        high = high_freq / nyquist
        order = 4
        b, a = butter(order, [low, high], btype='band')

        # Apply filter to each trial and channel
        for trial_idx in range(num_trials):
            for channel_idx in range(num_channels):
                filtered_data[trial_idx, band_idx, channel_idx, :] = filtfilt(
                    b, a, X[trial_idx, channel_idx, :]
                )

    return filtered_data


class TripletEEGDataset(Dataset):
    """
    Dataset for generating triplets (anchor, positive, negative) for EEG data.

    Parameters:
    -----------
    X : numpy.ndarray
        EEG data
    user_ids : numpy.ndarray
        User IDs corresponding to each EEG trial
    """

    def __init__(self, X, user_ids):
        self.X = X
        self.user_ids = user_ids
        self.user_to_indices = {}

        # Group indices by user
        for idx, user_id in enumerate(user_ids):
            if user_id not in self.user_to_indices:
                self.user_to_indices[user_id] = []
            self.user_to_indices[user_id].append(idx)

        # Only include users with at least 2 samples
        self.valid_users = [user for user, indices in self.user_to_indices.items()
                            if len(indices) >= 2]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """Get a triplet (anchor, positive, negative)"""
        anchor_idx = idx
        anchor_user = self.user_ids[anchor_idx]

        # Get positive (same user, different sample)
        user_samples = self.user_to_indices[anchor_user]
        positive_idx = np.random.choice([i for i in user_samples if i != anchor_idx])

        # Get negative (different user)
        negative_user = np.random.choice([u for u in self.valid_users if u != anchor_user])
        negative_idx = np.random.choice(self.user_to_indices[negative_user])

        return {
            'anchor': torch.tensor(self.X[anchor_idx], dtype=torch.float32),
            'positive': torch.tensor(self.X[positive_idx], dtype=torch.float32),
            'negative': torch.tensor(self.X[negative_idx], dtype=torch.float32)
        }


def create_data_loaders(data_dict, batch_size=64, num_workers=4):
    """
    Create data loaders for training, validation, and testing.

    Parameters:
    -----------
    data_dict : dict
        Dictionary containing EEG data and labels
    batch_size : int, default=64
        Batch size for data loaders
    num_workers : int, default=4
        Number of worker threads for data loading

    Returns:
    --------
    dict
        Dictionary containing data loaders for training, validation, and testing
    """
    train_triplet_dataset = TripletEEGDataset(data_dict['X_train'], data_dict['user_train'])
    val_triplet_dataset = TripletEEGDataset(data_dict['X_val'], data_dict['user_val'])

    train_triplet_loader = DataLoader(
        train_triplet_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_triplet_loader = DataLoader(
        val_triplet_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(data_dict['X_train'], dtype=torch.float32),
        torch.tensor(data_dict['user_train'], dtype=torch.long)
    )

    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(data_dict['X_val'], dtype=torch.float32),
        torch.tensor(data_dict['user_val'], dtype=torch.long)
    )

    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(data_dict['X_test'], dtype=torch.float32),
        torch.tensor(data_dict['user_test'], dtype=torch.long)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return {
        'triplet': {
            'train': train_triplet_loader,
            'val': val_triplet_loader
        },
        'classification': {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
    }

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = os.cpu_count()
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
  """
  # Use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
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
      shuffle=False, # don't need to shuffle test data
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names

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


def extract_embeddings(model, data_loader, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Extract embeddings from a trained EEG model.

    Parameters:
    -----------
    model : torch.nn.Module
        Trained EEGNet or FBCNet model
    data_loader : torch.utils.data.DataLoader
        DataLoader containing EEG data and labels
    device : str
        Device to run inference on ('cuda' or 'cpu')

    Returns:
    --------
    dict
        Dictionary containing embeddings, participant IDs, and labels
    """
    model.eval()
    model.to(device)

    all_embeddings = []
    all_participant_ids = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, dict):
                X = batch['anchor']
                participant_id = batch.get('participant_id', None)
                label = batch.get('label', None)
            elif len(batch) == 2:
                # Standard (X, y) format
                X, participant_id = batch
                label = None
            elif len(batch) == 3:
                X, participant_id, label = batch

            X = X.to(device)

            embeddings = model(X)

            all_embeddings.append(embeddings.cpu().numpy())

            if participant_id is not None:
                all_participant_ids.append(participant_id.cpu().numpy())
            if label is not None:
                all_labels.append(label.cpu().numpy())

    # Concatenate all batches
    embeddings = np.vstack(all_embeddings)

    if len(all_participant_ids) > 0:
        participant_ids = np.concatenate(all_participant_ids)
    else:
        participant_ids = np.array([0] * len(embeddings))

    if len(all_labels) > 0:
        labels = np.concatenate(all_labels)
    else:
        labels = np.array(["unknown"] * len(embeddings))

    return {
        'embeddings': embeddings,
        'participant_ids': participant_ids,
        'labels': labels
    }