from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from eeg_lib.utils.helpers import get_device

from typing import Optional
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter


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
