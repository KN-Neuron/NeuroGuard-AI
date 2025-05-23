import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
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

def plot_sample_signal(sample: pd.DataFrame):
    """
    Plot the sample signal from the first channel of the provided data.

    Parameters
    ----------
    sample : pd.DataFrame
        DataFrame containing the signal data to be plotted. The signal from
        the first column (channel) will be plotted.

    This function creates a plot of the signal over time for the first channel
    and displays it with appropriate labels for the axes and a title.
    """
    plt.figure(figsize=(15, 7))
    plt.plot(sample[0])
    plt.xlabel('Time')
    plt.title('Sample signal from first channel')
    plt.ylabel('Amplitude')
    plt.show()

def plot_sample_signals(sample: pd.DataFrame):
    """
    Plot the sample signals from all channels of the provided data.

    Parameters
    ----------
    sample : pd.DataFrame
        DataFrame containing the signal data to be plotted. Each column
        represents a different channel.

    This function creates a plot for signals from all channels over time
    and displays it with appropriate labels for the axes and a legend for
    channel identification.
    """
    plt.figure(figsize=(15, 7))
    for channel in sample:
        plt.plot(channel, alpha=0.5)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
    plt.title(f'Sample signal from all channels')
    plt.legend(['channel 1', 'channel 2', 'channel 3', 'channel 4'])
    plt.show()

def plot_first_channel(eeg_df: pd.DataFrame):
    """
    Plot the first channel signal from all participants in the provided DataFrame.

    Parameters
    ----------
    eeg_df : pd.DataFrame
        DataFrame containing the signal data to be plotted. Each row contains
        a single participant's data, and the 'epoch' column contains the signal
        data for that participant. The first column (channel) will be plotted.

    This function creates a plot for the first channel signal from all
    participants over time and displays it with appropriate labels for the
    axes and a title.
    """
    plt.figure(figsize=(15, 7))
    for signal in eeg_df['epoch'].iloc:
        plt.plot(signal[0], alpha=0.5)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
    plt.title(f'First channel signal from all participants')
    plt.show()

def plot_confusion_matrix(conf_matrix: np.ndarray):
    """
    Plot a confusion matrix using a heatmap.

    Parameters
    ----------
    conf_matrix : array-like
        The confusion matrix to be visualized.

    This function creates a heatmap representation of the given confusion matrix
    and displays it with labeled axes and a title.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion matrix")
    plt.show()

def plot_PSD(sample: pd.DataFrame):
    """
    Plot the Power Spectral Density (PSD) of signals from all channels in the provided sample.

    Parameters
    ----------
    sample : pd.DataFrame
        DataFrame containing the signal data to be analyzed. Each column
        represents a different channel.

    This function calculates and plots the PSD for each channel in the sample
    and displays it with appropriate labels for the axes and a legend for
    channel identification.
    """
    plt.figure(figsize=(15, 7))
    for channel in sample:
        f, Pxx_den = plt.psd(channel, NFFT=256, Fs=1.0, alpha=0.5)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB/Hz)')
    plt.title('Power Spectral Density (PSD) for all channels for sample')
    plt.legend(['channel 1', 'channel 2', 'channel 3', 'channel 4'])
    plt.show()
