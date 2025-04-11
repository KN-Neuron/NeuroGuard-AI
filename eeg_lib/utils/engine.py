from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve
from eeg_lib.utils.helpers import get_device

from typing import Optional
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import matplotlib.pyplot as plt


def verify_test_sample(test_df, model, user_profiles, sample_index=0, threshold=0.5):
    """
    Verifies a sample from the test dataframe against the user profiles.

    Args:
        test_df (DataFrame): The test dataframe containing EEG data.
        model (torch.nn.Module): The trained EEGNet model.
        user_profiles (dict): Dictionary of user profiles with participant IDs as keys.
        sample_index (int): Index of the sample to verify in the test dataframe.
        threshold (float): Distance threshold for verification.

    Returns:
        tuple: (accepted (bool), distance (float)) - Verification result and distance.
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


def verify_sample(new_embedding, user_profile, threshold):
    distance = np.linalg.norm(new_embedding - user_profile)
    return distance < threshold, distance


def create_user_profiles(embeddings_array, participant_ids_array):
    """
    Create user profiles by averaging embeddings for each participant.

    Args:
        embeddings_array (np.ndarray): Array of embeddings of shape (N, D), where N is the number of samples and D is the embedding dimension.
        participant_ids_array (np.ndarray): Array of participant IDs corresponding to the embeddings.

    Returns:
        dict: A dictionary where keys are participant IDs and values are the averaged embeddings (user profiles).
    """
    user_profiles = defaultdict(list)
    for embedding, pid in zip(embeddings_array, participant_ids_array):
        user_profiles[pid].append(embedding)

    for pid in user_profiles:
        user_profiles[pid] = np.mean(user_profiles[pid], axis=0)

    return user_profiles


def train_triplet_model(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn,
    optimizer,
    num_epochs: int,
    summary_writer: Optional[SummaryWriter] = None,
    device: str = get_device(),
):
    """
    Train the EEGNet model using triplet loss on triplet data.

    This function iterates over the provided training DataLoader, which yields triplets of
    EEG data (anchor, positive, negative). For each batch, the model generates embeddings for
    the three inputs, applies L2 normalization, computes the triplet margin loss, and updates
    the model parameters using the provided optimizer. Optionally, it logs the average loss per
    epoch and the model graph to TensorBoard via a SummaryWriter.

    Args:
        model (nn.Module): The neural network model (e.g., an EEGNetEmbeddingModel) to be trained.
        train_loader (DataLoader): A DataLoader that provides batches of triplet data in the form
            (anchor, positive, negative), where each sample is a tensor of shape (1, channels, time_points).
        loss_fn: The loss function to be used (e.g., nn.TripletMarginLoss) for computing the triplet loss.
        optimizer: The optimizer (e.g., torch.optim.Adam) used to update the model's parameters.
        num_epochs (int): The number of epochs for which to train the model.
        summary_writer (Optional[SummaryWriter], optional): A TensorBoard SummaryWriter instance for logging
            training metrics and the model graph. Defaults to None.
        device (str, optional): The device on which to run training (e.g., "cuda" or "cpu"). Defaults to the
            value returned by get_device().

    Returns:
        dict: A dictionary containing training metrics. Currently, it includes:
            - "avg_loss": A list of average losses for each epoch.

    Side Effects:
        - Updates the model parameters via backpropagation.
        - If a summary_writer is provided, logs the average loss per epoch and the model graph (on the first epoch).
        - Prints the average loss for each epoch.
    """
    model.train()
    results = {
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
            summary_writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={"avg_loss": avg_loss},
                global_step=epoch,
            )

        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

    return results, summary_writer


def extract_embeddings(model, test_df):
    """
    Extract embeddings and participant IDs from the test dataset using the given model.

    Args:
        model (torch.nn.Module): The trained model to generate embeddings.
        test_df (pd.DataFrame): The test dataset containing epochs and participant IDs.

    Returns:
        np.ndarray: Array of embeddings of shape (N, D), where N is the number of samples and D is the embedding dimension.
        np.ndarray: Array of participant IDs corresponding to the embeddings.
    """
    import numpy as np

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

    embeddings_array = np.stack(embeddings_list)  # (N, 32)
    participant_ids_array = np.array(participant_ids_list)

    return embeddings_array, participant_ids_array


def compute_pairwise_distances(embeddings_array, participant_ids_array):
    """

    Compute pairwise distances for genuine and imposter pairs.



    Args:

        embeddings_array (np.ndarray): Array of embeddings of shape (N, D).

        participant_ids_array (np.ndarray): Array of participant IDs.



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

    return genuine_distances, imposter_distances


def find_eer_threshold(genuine_distances, imposter_distances):
    """
    Find the Equal Error Rate (EER) threshold.

    The EER is the point where the False Acceptance Rate (FAR) equals the False Rejection Rate (FRR).

    Args:
        genuine_distances (list): List of distances for genuine pairs.
        imposter_distances (list): List of distances for imposter pairs.

    Returns:
        float: The threshold at which EER occurs.
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
    eer_threshold = thresholds[eer_index]

    return eer_threshold


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
