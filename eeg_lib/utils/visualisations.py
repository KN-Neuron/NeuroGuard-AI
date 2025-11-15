"""Visualization utilities for EEG data and embeddings."""
import matplotlib.pyplot as plt

import numpy as np
import numpy.typing as npt
import umap.umap_ as umap
from sklearn.manifold import TSNE  # type: ignore[import-untyped]
from typing import Any, Tuple, List, Optional


def calculate_and_plot_distances(
    embeddings_array: npt.NDArray[Any],
    participant_ids_array: npt.NDArray[Any],
    bins: int = 30
) -> Tuple[List[np.floating[Any]], List[np.floating[Any]]]:
    """
    Calculate and plot the distribution of pairwise distances for genuine and imposter pairs.

    Args:
        embeddings_array: Array of embeddings of shape (N, D), where N is the number of samples and D is the embedding dimension.
        participant_ids_array: Array of participant IDs corresponding to the embeddings.
        bins: Number of bins for the histogram.

    Returns:
        A tuple containing two lists - genuine_distances and imposter_distances.
    """
    genuine_distances = []
    imposter_distances = []

    N = len(embeddings_array)
    for i in range(N):
        for j in range(i + 1, N):
            distance = np.linalg.norm(embeddings_array[i] - embeddings_array[j])
            if participant_ids_array[i] == participant_ids_array[j]:
                genuine_distances.append(distance)
            else:
                imposter_distances.append(distance)

    plt.figure(figsize=(10, 6))
    plt.hist(genuine_distances, bins=bins, alpha=0.5, label="Genuine Pairs")
    plt.hist(imposter_distances, bins=bins, alpha=0.5, label="Imposter Pairs")
    plt.title("Distribution of Pairwise Distances")
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    return genuine_distances, imposter_distances


def visualize_tsne(
    embeddings: npt.NDArray[Any],
    participant_ids: npt.NDArray[Any],
    title: str = "t-SNE Visualization of EEG Embeddings"
) -> None:
    """
    Visualize embeddings using t-SNE.

    Args:
        embeddings: The embeddings array of shape (N, D), where N is the number of samples and D is the embedding dimension.
        participant_ids: Array of participant IDs corresponding to the embeddings.
        title: Title of the plot.
    """
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    unique_users = np.unique(participant_ids)
    for uid in unique_users:
        indices = np.where(participant_ids == uid)
        plt.scatter(
            embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=str(uid)
        )
    plt.legend(title="User ID")
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()


def plot_predictions(
    train_data: npt.NDArray[Any],
    train_labels: npt.NDArray[Any],
    test_data: npt.NDArray[Any],
    test_labels: npt.NDArray[Any],
    predictions: Optional[npt.NDArray[Any]] = None
) -> None:
    """
    Plots linear training data and test data and compares predictions.

    Args:
        train_data: Training input data
        train_labels: Training target labels
        test_data: Test input data
        test_labels: Test target labels
        predictions: Predictions to compare against test data
    """
    plt.figure(figsize=(10, 7))

    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})


def visualize_umap(
    embeddings: npt.NDArray[Any],
    participant_ids: npt.NDArray[Any],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    title: str = "UMAP Visualization of EEG Embeddings",
) -> None:
    """
    Visualize embeddings using UMAP.

    Args:
        embeddings: The embeddings array of shape (N, D), where N is the number of samples and D is the embedding dimension.
        participant_ids: Array of participant IDs corresponding to the embeddings.
        n_neighbors: The size of the local neighborhood (in terms of number of neighboring points) used for manifold approximation.
        min_dist: The effective minimum distance between embedded points.
        random_state: Random seed for reproducibility.
        title: Title of the plot.
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state
    )
    umap_coords = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    unique_users = np.unique(participant_ids)
    for user in unique_users:
        indices = np.where(participant_ids == user)
        plt.scatter(umap_coords[indices, 0], umap_coords[indices, 1], label=str(user))

    plt.legend(title="User ID")
    plt.title(title)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.show()
