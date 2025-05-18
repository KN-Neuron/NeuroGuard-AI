import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import umap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA


def plot_loss(train_history: dict):
    """Plot training curves with proper metrics for authentication systems"""

    plt.figure(figsize=(15, 5))

    plt.plot(train_history["train_loss"], label="Train Loss")
    plt.title("Triplet Margin Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_embeddings_2d(embeddings_2d: np.ndarray):
    """
    Plot 2D visualization of embeddings

    Parameters
    ----------
    embeddings_2d : np.ndarray
        2D embeddings of data points
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=10, c='blue')
    plt.title("2D Visualization of Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.show()


def plot_embeddings_by_participant(embeddings_2d: np.ndarray, participant_ids: np.ndarray):
    """
    Plot 2D visualization of embeddings by participant

    Parameters
    ----------
    embeddings_2d : np.ndarray
        2D embeddings of data points
    participant_ids : np.ndarray
        Array of participant ids
    """
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


def plot_UMAP(embeddings_2d: np.ndarray, participant_ids: np.ndarray):
    """
    Plot UMAP visualization of embeddings by participant

    Parameters
    ----------
    embeddings_2d : np.ndarray
        2D embeddings of data points
    participant_ids : np.ndarray
        Array of participant ids
    """
    unique_participants = np.unique(participant_ids)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_participants)))

    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    umap_embeddings = umap_model.fit_transform(embeddings_2d)

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




def plot_PCA(test_embeddings: np.ndarray, participant_ids: np.ndarray):
    """
    Plot PCA visualization of embeddings by participant

    Parameters
    ----------
    test_embeddings : np.ndarray
        2D embeddings of data points
    participant_ids : np.ndarray
        Array of participant ids
    """
    unique_participants = np.unique(participant_ids)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_participants)))

    pca_test = PCA(n_components=2)
    pca_test_embeddings = pca_test.fit_transform(test_embeddings)

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


def plot_tSNE(test_embeddings: np.ndarray, participant_ids: np.ndarray):
    """
    Plot t-SNE visualization of embeddings by participant

    Parameters
    ----------
    test_embeddings : np.ndarray
        2D embeddings of data points
    participant_ids : np.ndarray
        Array of participant ids
    """
    unique_participants = np.unique(participant_ids)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_participants)))

    tsne_test = TSNE(n_components=2)
    tsne_test_embeddings = tsne_test.fit_transform(test_embeddings)

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


def plot_LDA(test_embeddings: np.ndarray, participant_ids: np.ndarray):
    """
    Plot LDA visualization of embeddings by participant

    Parameters
    ----------
    test_embeddings : np.ndarray
        2D embeddings of data points
    participant_ids : np.ndarray
        Array of participant ids
    """
    unique_participants = np.unique(participant_ids)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_participants)))

    lda_test = LinearDiscriminantAnalysis(n_components=2)
    lda_test_embeddings = lda_test.fit_transform(test_embeddings, participant_ids)

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