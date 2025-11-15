"""Training and evaluation engine for EEG models."""
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve
from eeg_lib.utils.helpers import get_device
from eeg_lib.models.verification.EEGNet import EEGNetEmbeddingModel
from eeg_lib.types import EEGDataTensor, ModelOutputTensor

from typing import Optional, Tuple, Dict, Any, List
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
import numpy.typing as npt
import torch
import pandas as pd
from torch.utils.tensorboard.writer import SummaryWriter
import matplotlib.pyplot as plt


def verify_test_sample(
    test_df: pd.DataFrame,
    model: EEGNetEmbeddingModel,
    user_profiles: Dict[str, npt.NDArray[Any]],
    sample_index: int = 0,
    threshold: float = 0.5
) -> Tuple[bool, float]:
    """
    Verifies a sample from the test dataframe against the user profiles.

    Args:
        test_df: The test dataframe containing EEG data.
        model: The trained EEGNet model.
        user_profiles: Dictionary of user profiles with participant IDs as keys.
        sample_index: Index of the sample to verify in the test dataframe.
        threshold: Distance threshold for verification.

    Returns:
        (accepted, distance) - Verification result and distance.
    """
    sample_data = (
        torch.tensor(test_df.iloc[sample_index]["epoch"], dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    with torch.no_grad():
        sample_embedding, _ = model(sample_data)
        sample_embedding = (
            F.normalize(sample_embedding, p=2, dim=1).squeeze(0).cpu().numpy()
        )

    user_id = test_df.iloc[sample_index]["participant_id"]
    user_profile = user_profiles[user_id]

    accepted, distance = verify_sample(sample_embedding, user_profile, threshold)
    return accepted, distance


def verify_sample(
    new_embedding: npt.NDArray[Any],
    user_profile: npt.NDArray[Any],
    threshold: float
) -> Tuple[bool, float]:
    """
    Verifies a new embedding against a user profile.

    Args:
        new_embedding: The embedding to verify
        user_profile: The reference user profile embedding
        threshold: Distance threshold for verification

    Returns:
        (accepted, distance) - Verification result and distance
    """
    distance = np.linalg.norm(new_embedding - user_profile)
    return bool(distance < threshold), float(distance)


def create_user_profiles(
    embeddings_array: npt.NDArray[Any],
    participant_ids_array: npt.NDArray[Any]
) -> Dict[str, npt.NDArray[Any]]:
    """
    Create user profiles by averaging embeddings for each participant.

    Args:
        embeddings_array: Array of embeddings of shape (N, D), where N is the number of samples and D is the embedding dimension.
        participant_ids_array: Array of participant IDs corresponding to the embeddings.

    Returns:
        A dictionary where keys are participant IDs and values are the averaged embeddings (user profiles).
    """
    user_profiles: Dict[str, List[npt.NDArray[Any]]] = defaultdict(list)
    for embedding, pid in zip(embeddings_array, participant_ids_array):
        user_profiles[str(pid)].append(embedding)

    averaged_profiles = {}
    for pid, embeddings_list in user_profiles.items():
        averaged_profiles[pid] = np.mean(embeddings_list, axis=0)

    return averaged_profiles


def train_triplet_model(
    model: nn.Module,
    train_loader: DataLoader[Any],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    summary_writer: Optional[SummaryWriter] = None,
    device: str = get_device(),
) -> Tuple[Dict[str, List[float]], Optional[SummaryWriter]]:
    """
    Train the EEGNet model using triplet loss on triplet data.

    This function iterates over the provided training DataLoader, which yields triplets of
    EEG data (anchor, positive, negative). For each batch, the model generates embeddings for
    the three inputs, applies L2 normalization, computes the triplet margin loss, and updates
    the model parameters using the provided optimizer. Optionally, it logs the average loss per
    epoch and the model graph to TensorBoard via a SummaryWriter.

    Args:
        model: The neural network model (e.g., an EEGNetEmbeddingModel) to be trained.
        train_loader: A DataLoader that provides batches of triplet data in the form
            (anchor, positive, negative), where each sample is a tensor of shape (1, channels, time_points).
        loss_fn: The loss function to be used (e.g., nn.TripletMarginLoss) for computing the triplet loss.
        optimizer: The optimizer (e.g., torch.optim.Adam) used to update the model's parameters.
        num_epochs: The number of epochs for which to train the model.
        summary_writer: A TensorBoard SummaryWriter instance for logging
            training metrics and the model graph. Defaults to None.
        device: The device on which to run training (e.g., "cuda" or "cpu"). Defaults to the
            value returned by get_device().

    Returns:
        A tuple containing:
            - dict: A dictionary containing training metrics. Currently, it includes:
                - "avg_loss": A list of average losses for each epoch.
            - Optional[SummaryWriter]: The summary writer (possibly updated)

    Side Effects:
        - Updates the model parameters via backpropagation.
        - If a summary_writer is provided, logs the average loss per epoch and the model graph (on the first epoch).
        - Prints the average loss for each epoch.
    """
    model.train()
    results: Dict[str, List[float]] = {
        "avg_loss": [],
    }
    for epoch in range(num_epochs):
        running_loss = 0.0

        for anchor, positive, negative in train_loader:
            anchor_embedding, _ = model(anchor)
            pos_embedding, _ = model(positive)
            neg_embedding, _ = model(negative)

            anchor_embedding = F.normalize(anchor_embedding, p=2, dim=1)
            pos_embedding = F.normalize(pos_embedding, p=2, dim=1)
            neg_embedding = F.normalize(neg_embedding, p=2, dim=1)

            loss = loss_fn(anchor_embedding, pos_embedding, neg_embedding)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        results["avg_loss"].append(avg_loss)
        if summary_writer:
            summary_writer.add_scalars(  # type: ignore[no-untyped-call]
                main_tag="Loss",
                tag_scalar_dict={"avg_loss": avg_loss},
                global_step=epoch,
            )

        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

    return results, summary_writer


def extract_embeddings(
    model: EEGNetEmbeddingModel,
    test_df: pd.DataFrame
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """
    Extract embeddings and participant IDs from the test dataset using the given model.

    Args:
        model: The trained model to generate embeddings.
        test_df: The test dataset containing epochs and participant IDs.

    Returns:
        A tuple containing:
            - Array of embeddings of shape (N, D), where N is the number of samples and D is the embedding dimension.
            - Array of participant IDs corresponding to the embeddings.
    """
    model.eval()
    embeddings_list = []
    participant_ids_list = []

    with torch.no_grad():
        for _, row in test_df.iterrows():
            # Each row's epoch is of shape (4, 751); add channel dim → (1,4,751)
            data = (
                torch.tensor(row["epoch"], dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            embedding, _ = model(data)
            embedding = F.normalize(
                embedding, p=2, dim=1
            )  # normalize for similarity metrics
            embeddings_list.append(embedding.squeeze(0).cpu().numpy())
            participant_ids_list.append(row["participant_id"])

    embeddings_array = np.stack(embeddings_list)  # (N, D) where D=embedding dimension
    participant_ids_array = np.array(participant_ids_list)

    return embeddings_array, participant_ids_array


def compute_pairwise_distances(
    embeddings_array: npt.NDArray[Any],
    participant_ids_array: npt.NDArray[Any]
) -> Tuple[List[float], List[float]]:
    """
    Compute pairwise distances for genuine and imposter pairs.

    Args:
        embeddings_array: Array of embeddings of shape (N, D).
        participant_ids_array: Array of participant IDs.

    Returns:
        A tuple containing two lists - genuine_distances and imposter_distances.
    """
    genuine_distances: List[float] = []
    imposter_distances: List[float] = []

    N = len(embeddings_array)

    for i in range(N):
        for j in range(i + 1, N):
            distance = float(np.linalg.norm(embeddings_array[i] - embeddings_array[j]))

            if participant_ids_array[i] == participant_ids_array[j]:
                genuine_distances.append(distance)
            else:
                imposter_distances.append(distance)

    return genuine_distances, imposter_distances


def find_eer_threshold(
    genuine_distances: List[float],
    imposter_distances: List[float]
) -> float:
    """
    Find the Equal Error Rate (EER) threshold.

    The EER is the point where the False Acceptance Rate (FAR) equals the False Rejection Rate (FRR).

    Args:
        genuine_distances: List of distances for genuine pairs.
        imposter_distances: List of distances for imposter pairs.

    Returns:
        The threshold at which EER occurs.
    """
    distances = np.concatenate([genuine_distances, imposter_distances])
    labels = np.concatenate(
        [np.ones(len(genuine_distances)), np.zeros(len(imposter_distances))]
    )

    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels, distances, pos_label=0
    )

    false_rejection_rate = 1 - true_positive_rate

    eer_index = np.nanargmin(np.abs(false_positive_rate - false_rejection_rate))
    eer_threshold = float(thresholds[eer_index])

    return eer_threshold


def plot_2d_embeddings(
    embeddings_2d: npt.NDArray[Any],
    participant_ids: npt.NDArray[Any],
    method_name: str = "t-SNE"
) -> None:
    """
    Plots 2D embeddings colored by participant/user ID.

    Args:
        embeddings_2d: 2D array of shape (N, 2).
        participant_ids: Array of shape (N,) with user IDs.
        method_name: Name of the dimensionality reduction method (for title).
    """
    plt.figure(figsize=(10, 8))
    unique_users = np.unique(participant_ids)
    for user in unique_users:
        idx = np.where(participant_ids == user)
        plt.scatter(
            embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=str(user), alpha=0.7
        )
    plt.title(f"{method_name} Visualization of EEG Embeddings (All Users)")
    plt.xlabel(f"{method_name} Dimension 1")
    plt.ylabel(f"{method_name} Dimension 2")
    plt.legend(title="User ID", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def run_threshold_and_visualization(
    model: EEGNetEmbeddingModel,
    test_df: pd.DataFrame
) -> Tuple[float, float, float, float, npt.NDArray[Any]]:
    """
    Extracts embeddings for all test samples, performs threshold selection by computing the
    genuine and imposter pairwise distances, and then produces t-SNE plots for all users.

    Args:
        model: Trained EEGNet model.
        test_df: Test DataFrame containing columns 'participant_id' and 'epoch'.

    Returns:
        A tuple containing:
            - best_threshold: The selected threshold (approximate EER point).
            - approximate_eer: The computed equal error rate.
            - final_FRR: Final false reject rate at the threshold.
            - final_FAR: Final false accept rate at the threshold.
            - tsne_coords: 2D coordinates from t-SNE.
    """
    embeddings_array, participant_ids_array = extract_embeddings(model, test_df)
    print(f"Extracted embeddings shape: {embeddings_array.shape}")

    genuine_distances, imposter_distances = compute_pairwise_distances(
        embeddings_array, participant_ids_array
    )

    best_threshold = find_eer_threshold(
        genuine_distances, imposter_distances
    )
    print(f"Selected threshold = {best_threshold:.4f}")

    # Compute approximate EER, FRR, and FAR at the threshold
    # This is a simplified calculation - you might want to implement more rigorous metrics
    thresholded_genuine = [d for d in genuine_distances if d <= best_threshold]
    thresholded_imposter = [d for d in imposter_distances if d <= best_threshold]

    far = len(thresholded_imposter) / len(imposter_distances) if len(imposter_distances) > 0 else 0.0
    frr = len([d for d in genuine_distances if d > best_threshold]) / len(genuine_distances) if len(genuine_distances) > 0 else 0.0
    approximate_eer = (far + frr) / 2.0

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_coords = tsne.fit_transform(embeddings_array)
    plot_2d_embeddings(tsne_coords, participant_ids_array, method_name="t-SNE")

    # Note: UMAP implementation is commented out in the original code
    # To use it, uncomment and install umap-learn: pip install umap-learn
    # reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    # umap_coords = reducer.fit_transform(embeddings_array)
    # plot_2d_embeddings(umap_coords, participant_ids_array, method_name="UMAP")

    return best_threshold, approximate_eer, frr, far, tsne_coords
