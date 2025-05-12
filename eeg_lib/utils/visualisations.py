import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import pandas as pd


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


def plot_loss_curves(results: dict):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


def plot_loss(train_history):
    """Plot training curves with proper metrics for authentication systems"""

    plt.figure(figsize=(15, 5))

    plt.plot(train_history["train_loss"], label="Train Loss")
    plt.title("Triplet Margin Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_embeddings_2d(embeddings_2d):
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=10, c='blue')
    plt.title("2D Visualization of Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.show()


def plot_embeddings_by_participant(embeddings_2d, participant_ids):
    unique_participants = np.unique(participant_ids)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_participants)))

    plt.figure(figsize=(10, 8))

    for i, participant in enumerate(unique_participants):
        mask = participant_ids == participant
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=participant,
            color=colors[i],
            alpha=0.6,
            s=10
        )

    plt.title("2D Visualization of Embeddings by Participant")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


import umap


def plot_UMAP(embeddings_2d, participant_ids):
    unique_participants = np.unique(participant_ids)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_participants)))

    # Generate UMAP embeddings
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    umap_embeddings = umap_model.fit_transform(embeddings_2d)

    # Plot the UMAP embeddings
    plt.figure(figsize=(10, 8))
    for i, participant in enumerate(unique_participants):
        mask = participant_ids == participant
        plt.scatter(
            umap_embeddings[mask, 0],
            umap_embeddings[mask, 1],
            label=participant,
            color=colors[i],
            alpha=0.6,
            s=10
        )

    plt.title("UMAP Visualization of Embeddings by Participant")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


from sklearn.decomposition import PCA


def plot_PCA(test_embeddings, participant_ids):
    unique_participants = np.unique(participant_ids)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_participants)))

    # Apply PCA to reduce dimensions to 2
    pca_test = PCA(n_components=2)
    pca_test_embeddings = pca_test.fit_transform(test_embeddings)

    # Plot the PCA embeddings for test data
    plt.figure(figsize=(10, 8))
    for i, participant in enumerate(unique_participants):
        mask = participant_ids == participant
        plt.scatter(
            pca_test_embeddings[mask, 0],
            pca_test_embeddings[mask, 1],
            label=participant,
            color=colors[i],
            alpha=0.6,
            s=10
        )

    plt.title("PCA Visualization of Test Participants")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_tSNE(test_embeddings, participant_ids):
    unique_participants = np.unique(participant_ids)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_participants)))

    # Apply t-SNE to reduce dimensions to 2
    tsne_test = TSNE(n_components=2)
    tsne_test_embeddings = tsne_test.fit_transform(test_embeddings)

    # Plot the t-SNE embeddings for test data
    plt.figure(figsize=(10, 8))
    for i, participant in enumerate(unique_participants):
        mask = participant_ids == participant
        plt.scatter(
            tsne_test_embeddings[mask, 0],
            tsne_test_embeddings[mask, 1],
            label=participant,
            color=colors[i],
            alpha=0.6,
            s=10
        )

    plt.title("t-SNE Visualization of Test Participants")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
def plot_LDA(test_embeddings, participant_ids):
    unique_participants = np.unique(participant_ids)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_participants)))

    lda_test = LinearDiscriminantAnalysis(n_components=2)
    lda_test_embeddings = lda_test.fit_transform(test_embeddings, participant_ids)

    # Plot the LDA embeddings for test data
    plt.figure(figsize=(10, 8))

    for i, participant in enumerate(unique_participants):
        mask = participant_ids == participant
        plt.scatter(
            lda_test_embeddings[mask, 0],
            lda_test_embeddings[mask, 1],
            label=participant,
            color=colors[i],
            alpha=0.6,
            s=10
        )

    plt.title("LDA Visualization of Test Participants")
    plt.xlabel("LDA Dimension 1")
    plt.ylabel("LDA Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.show()