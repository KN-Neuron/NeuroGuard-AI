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
    random_state: int = 0,
    ret_colors: bool = False,
) -> tuple[Any, Any, Any, Any, Any, Any] | tuple[Any, Any, Any, Any]:
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

    train_df = eeg_df[eeg_df["participant_id"].isin(train_participants)].reset_index(
        drop=True
    )
    test_df = eeg_df[eeg_df["participant_id"].isin(test_participants)].reset_index(
        drop=True
    )

    y_train = train_df["participant_id"].values

    y_test = test_df["participant_id"].values
    X_train = train_df["epoch"].values
    X_test = test_df["epoch"].values
    colors_train = None
    colors_test = None
    if ret_colors:
        colors_train = train_df["label"].values
        colors_test = test_df["label"].values

    print("Training set participants:", train_participants)
    print("Test set participants:", test_participants)
    print("Training labels:", np.unique(y_train))
    print("Test labels:", np.unique(y_test))

    if ret_colors:
        return X_train, X_test, y_train, y_test, colors_train, colors_test

    return X_train, X_test, y_train, y_test


def compute_genuine_imposter_distances(
    embeddings: np.ndarray,
    ids: np.ndarray,
    user_profiles: dict,
    distance_metric: str = "euclidean",
) -> (np.ndarray, np.ndarray):
    """
    Given embeddings and their user IDs, compute:
      - genuine_dists: distance(emb_i, profile[user_i])
      - imposter_dists: distance(emb_i, profile[user_j])

    Args:
        embeddings (np.ndarray): shape = (N, D). N - num samples, D - embedding dimension.
        ids (np.ndarray): shape = (N,), integer or string IDs of user profiles
        user_profiles (dict): { user_id: mean_embedding (D,) } from training set.
        distance_metric (str): "euclidean" or "cosine".

    Returns:
        genuine_dists (np.ndarray): shape = (N,)
        imposter_dists (np.ndarray): shape = (N * (num_users-1),)
    """
    N, D = embeddings.shape
    unique_pids = list(user_profiles.keys())

    genuine_dists = np.zeros(N, dtype=float)
    imposter_dists_list = []

    profile_matrix = np.vstack([user_profiles[pid] for pid in unique_pids])
    pid_to_index = {pid: idx for idx, pid in enumerate(unique_pids)}

    all_dist_to_profiles = cdist(embeddings, profile_matrix, metric=distance_metric)

    for i in range(N):
        pid_i = ids[i]
        idx_i = pid_to_index[pid_i]
        genuine_dists[i] = all_dist_to_profiles[i, idx_i]

        for j in range(len(unique_pids)):
            if j == idx_i:
                continue
            imposter_dists_list.append(all_dist_to_profiles[i, j])

    return genuine_dists, np.array(imposter_dists_list)


def compute_threshold_metrics(
    genuine_dists: np.ndarray,
    imposter_dists: np.ndarray,
    num_thresholds: int = 200,
    eps: float = 0.0001,
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float):
    """
    Given arrays of genuine/imposter distances, sweep a range of thresholds
    and compute FNR, FPR, and Accuracy. Also identify the threshold
    at Equal Error Rate (EER) where FNR = FPR.

    Args:
        outputs of compute_genuine_imposter_distances:
        genuine_dists (np.ndarray): shape = (N_genuine,)
        imposter_dists (np.ndarray): shape = (N_imposter,)

        num_thresholds (int): how many evenly‐spaced thresholds to try.

    Returns:
        thresholds      (np.ndarray): shape = (num_thresholds,)
        fnr_list        (np.ndarray): shape = (num_thresholds,)
        fpr_list        (np.ndarray): shape = (num_thresholds,)
        acc_list        (np.ndarray): shape = (num_thresholds,)
        eer_threshold   (float): threshold at Equal Error Rate (EER)
        eer_fnr         (float): FNR at EER point
        eer_fpr         (float): FPR at EER point
        eer_acc         (float): Accuracy at EER point
    """
    num_genuine = float(len(genuine_dists))
    num_imposter = float(len(imposter_dists))
    total_pairs = num_genuine + num_imposter

    all_distances = np.concatenate([genuine_dists, imposter_dists])
    t_min = np.min(all_distances)
    t_max = np.max(all_distances)
    thresholds = np.linspace(t_min, t_max, num_thresholds)

    fnr_list = np.zeros(num_thresholds, dtype=float)
    fpr_list = np.zeros(num_thresholds, dtype=float)
    acc_list = np.zeros(num_thresholds, dtype=float)

    for idx, T in enumerate(thresholds):
        false_rejects = np.sum(genuine_dists > T)
        fnr = false_rejects / (num_genuine + eps)

        false_accepts = np.sum(imposter_dists <= T)
        fpr = false_accepts / (num_imposter + eps)

        true_accepts = num_genuine - false_rejects
        true_rejects = num_imposter - false_accepts
        accuracy = (true_accepts + true_rejects) / total_pairs

        fnr_list[idx] = fnr
        fpr_list[idx] = fpr
        acc_list[idx] = accuracy

    eer_diff = np.abs(fnr_list - fpr_list)
    eer_idx = np.argmin(eer_diff)

    eer_threshold = thresholds[eer_idx]
    eer_fnr = fnr_list[eer_idx]
    eer_fpr = fpr_list[eer_idx]
    eer_acc = acc_list[eer_idx]

    return (
        thresholds,
        fnr_list,
        fpr_list,
        acc_list,
        eer_threshold,
        eer_fnr,
        eer_fpr,
        eer_acc,
    )


def compute_f1_vs_threshold(
    genuine_dists: np.ndarray, imposter_dists: np.ndarray, num_thresholds: int = 200
) -> (np.ndarray, np.ndarray, float, float):
    """
    compute F1-score for a range of thresholds T:
      - "positive" = genuine pair  (label = 1)
      - "negative" = imposter pair (label = 0)

    Returns:
      thresholds (shape = (num_thresholds,)),
      f1_list    (shape = (num_thresholds,)),

      best_threshold (float),
      best_f1        (float)
    """
    # Number of genuine/imposter pairs:
    Pg = float(len(genuine_dists))
    Pi = float(len(imposter_dists))

    all_d = np.concatenate([genuine_dists, imposter_dists])
    t_min, t_max = np.min(all_d), np.max(all_d)
    thresholds = np.linspace(t_min, t_max, num_thresholds)

    f1_list = np.zeros(num_thresholds, dtype=float)

    for idx, T in enumerate(thresholds):
        TP = np.sum(genuine_dists <= T)
        FN = np.sum(genuine_dists > T)

        FP = np.sum(imposter_dists <= T)
        TN = np.sum(imposter_dists > T)

        if (TP + FP) > 0:
            precision = TP / float(TP + FP)
        else:
            precision = 0.0
        if (TP + FN) > 0:
            recall = TP / float(TP + FN)
        else:
            recall = 0.0

        if precision + recall > 0:
            f1 = 2.0 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        f1_list[idx] = f1

    best_idx = np.argmax(f1_list)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_list[best_idx]

    return thresholds, f1_list, best_threshold, best_f1


def split_test_data_for_verification(
    test_embeddings: np.ndarray, test_ids: np.ndarray, profile_ratio: float = 0.6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split test data into profile creation and verification portions.

    Parameters
    ----------
    test_embeddings : np.ndarray
        An array of embedding vectors for the test set, with a shape of
        (n_samples, embedding_dim).
    test_ids : np.ndarray
        An array of corresponding subject IDs for each embedding in
        `test_embeddings`, with a shape of (n_samples,).
    profile_ratio : float, optional
        The proportion of each subject's samples to be allocated to the
        profile set. The rest will be used for verification. Must be between
        0 and 1. Defaults to 0.6.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing four NumPy arrays:
        - profile_embeddings: Embeddings designated for creating user profiles.
        - profile_ids: Corresponding subject IDs for the profile embeddings.
        - verify_embeddings: Embeddings designated for verification attempts.
        - verify_ids: Corresponding subject IDs for the verification embeddings.
    """
    unique_subjects = np.unique(test_ids)

    profile_embeddings_list = []
    profile_ids_list = []
    verify_embeddings_list = []
    verify_ids_list = []

    for subject_id in unique_subjects:
        subject_mask = test_ids == subject_id
        subject_embeddings = test_embeddings[subject_mask]
        subject_samples = len(subject_embeddings)

        if subject_samples < 2:
            continue

        profile_samples = max(1, int(subject_samples * profile_ratio))

        profile_embeddings_list.extend(subject_embeddings[:profile_samples])
        profile_ids_list.extend([subject_id] * profile_samples)

        verify_embeddings_list.extend(subject_embeddings[profile_samples:])
        verify_ids_list.extend([subject_id] * (subject_samples - profile_samples))

    return (
        np.array(profile_embeddings_list),
        np.array(profile_ids_list),
        np.array(verify_embeddings_list),
        np.array(verify_ids_list),
    )
