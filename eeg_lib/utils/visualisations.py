import matplotlib.pyplot as plt

import numpy as np
import umap.umap_ as umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from eeg_lib.utils.engine import (
    compute_pairwise_distances,
    extract_embeddings,
    find_eer_threshold,
)


def calculate_and_plot_distances(embeddings_array, participant_ids_array, bins=30):
    """
    Calculate and plot the distribution of pairwise distances for genuine and imposter pairs.

    Args:
        embeddings_array (np.ndarray): Array of embeddings of shape (N, D), where N is the number of samples and D is the embedding dimension.
        participant_ids_array (np.ndarray): Array of participant IDs corresponding to the embeddings.
        bins (int): Number of bins for the histogram.

    Returns:
        tuple: A tuple containing two lists - genuine_distances and imposter_distances.
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
    embeddings, participant_ids, title="t-SNE Visualization of EEG Embeddings"
):
    """
    Visualize embeddings using t-SNE.

    Args:
        embeddings (np.ndarray): The embeddings array of shape (N, D), where N is the number of samples and D is the embedding dimension.
        participant_ids (np.ndarray): Array of participant IDs corresponding to the embeddings.
        title (str): Title of the plot.

    Returns:
        None
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
    train_data, train_labels, test_data, test_labels, predictions=None
):
    """
    Plots linear training data and test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))

    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})


def visualize_umap(
    embeddings,
    participant_ids,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42,
    title="UMAP Visualization of EEG Embeddings",
):
    """
    Visualize embeddings using UMAP.

    Args:
        embeddings (np.ndarray): The embeddings array of shape (N, D), where N is the number of samples and D is the embedding dimension.
        participant_ids (np.ndarray): Array of participant IDs corresponding to the embeddings.
        n_neighbors (int): The size of the local neighborhood (in terms of number of neighboring points) used for manifold approximation.
        min_dist (float): The effective minimum distance between embedded points.
        random_state (int): Random seed for reproducibility.
        title (str): Title of the plot.

    Returns:
        None
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


def plot_2d_embeddings(embeddings_2d, participant_ids, method_name="t-SNE"):
    """
    Plots 2D embeddings colored by participant/user ID.

    Args:
        embeddings_2d (np.ndarray): 2D array of shape (N, 2).
        participant_ids (np.ndarray): Array of shape (N,) with user IDs.
        method_name (str): Name of the dimensionality reduction method (for title).
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


def run_threshold_and_visualization(model, test_df):
    """
    Extracts embeddings for all test samples, performs threshold selection by computing the
    genuine and imposter pairwise distances, and then produces t-SNE and UMAP plots for all users.

    Args:
        model (nn.Module): Trained EEGNet model.
        test_df (pd.DataFrame): Test DataFrame containing columns 'participant_id' and 'epoch'.
        device (str): Device to perform inference on.

    Returns:
        Tuple[float, float, float, float, np.ndarray, np.ndarray]:
            - best_threshold: The selected threshold (approximate EER point).
            - approximate_eer: The computed equal error rate.
            - final_FRR: Final false reject rate at the threshold.
            - final_FAR: Final false accept rate at the threshold.
            - tsne_coords: 2D coordinates from t-SNE.
            - umap_coords: 2D coordinates from UMAP.
    """
    embeddings_array, participant_ids_array = extract_embeddings(model, test_df)
    print(f"Extracted embeddings shape: {embeddings_array.shape}")

    genuine_distances, imposter_distances = compute_pairwise_distances(
        embeddings_array, participant_ids_array
    )

    best_threshold, approximate_eer, final_FRR, final_FAR = find_eer_threshold(
        genuine_distances, imposter_distances
    )
    print(
        f"Selected threshold = {best_threshold:.4f}, EER = {approximate_eer:.4f} (FRR = {final_FRR:.4f}, FAR = {final_FAR:.4f})"
    )

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_coords = tsne.fit_transform(embeddings_array)
    plot_2d_embeddings(tsne_coords, participant_ids_array, method_name="t-SNE")

    # reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    # umap_coords = reducer.fit_transform(embeddings_array)
    # plot_2d_embeddings(umap_coords, participant_ids_array, method_name="UMAP")

    return best_threshold, approximate_eer, final_FRR, final_FAR, tsne_coords
