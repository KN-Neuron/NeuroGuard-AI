import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as mpatches
from typing import Optional, List, Tuple, Any, Union
from matplotlib.colors import Colormap
from scipy.spatial.distance import cdist

import numpy as np
import numpy.typing as npt
from sklearn.manifold import TSNE
from typing import Any, Tuple, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import umap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA


def calculate_and_plot_distances(
    embeddings_array: npt.NDArray[Any],
    participant_ids_array: npt.NDArray[Any],
    bins: int = 30,
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
    title: str = "t-SNE Visualization of EEG Embeddings",
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
    predictions: Optional[npt.NDArray[Any]] = None,
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


def plot_tsne(
    embeddings: np.ndarray,
    cmap: Colormap,
    labels: np.ndarray,
    handles: Optional[list[mpatches.Patch]] = None,
    figsize: tuple[int, int] = (10, 8),
    alpha: float = 1.0,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    centroids: Optional[dict] = None,
    test_embeddings: Optional[np.ndarray] = None,
    test_labels: Optional[np.ndarray] = None,
    save: bool = False,
    return_fig: bool = True,
) -> Optional[plt.Figure]:
    """
    Plots a 2D t-SNE visualization for given embeddings along with optional centroids and test data.

    The function creates a matplotlib figure displaying a scatter plot of the t-SNE
    embeddings. It uses a specified colormap and assigns colors based on the provided label values.
    Optionally, it can also plot centroids and test embeddings. A legend is created from the provided
    handles (if any); otherwise, a default legend is attempted.

    Args:
        embeddings (np.ndarray): A 2D numpy array of shape (N, 2) representing t-SNE embeddings for the data.
        cmap (Union[str, Colormap]): The colormap to use for the scatter plot.
        labels (np.ndarray): A 1D numpy array of numerical labels corresponding to the embeddings.
        handles (Optional[List[mpatches.Patch]], optional): A list of legend handles (e.g., patches)
            to use for the plot legend. Defaults to None.
        figsize (Tuple[int, int], optional): Figure size in inches as a tuple (width, height). Defaults to (10, 8).
        alpha (float, optional): The transparency value for the scatter points. Defaults to 1.0.
        title (Optional[str], optional): The title of the plot. Defaults to None.
        xlabel (Optional[str], optional): Label for the x-axis. Defaults to None.
        ylabel (Optional[str], optional): Label for the y-axis. Defaults to None.
        centroids (Optional[dict], optional): A dictionary mapping identifiers to 2D coordinates for centroids.
            If provided, these centroids are plotted with a distinct marker ("X"). Defaults to None.
        test_embeddings (Optional[np.ndarray], optional): A 2D numpy array of test embeddings to plot in addition.
            Defaults to None.
        test_labels (Optional[np.ndarray], optional): A 1D numpy array of labels corresponding to test_embeddings.
            Defaults to None.
        save (bool, optional): If True and a title is provided, the plot is saved as a PNG file in an 'images' folder.
            Defaults to False.
        return_fig (bool, optional): If True, the function returns the matplotlib Figure object. If False, the plot is shown and None is returned.
            Defaults to True.

    Returns:
        Optional[plt.Figure]: The matplotlib Figure object if return_fig is True; otherwise, None.

    Example:
        >>> fig = plot_tsne(
        ...     embeddings=train_reduced_normalized,
        ...     cmap='tab10',
        ...     labels=y_train_encoded,
        ...     handles=train_handles,
        ...     alpha=0.7,
        ...     title="t-SNE Visualization - Train Data",
        ...     xlabel="t-SNE Component 1",
        ...     ylabel="t-SNE Component 2"
        ... )
        >>> if fig:
        ...     fig.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    scatter_train = ax.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=labels,
        cmap=cmap,
        alpha=alpha,
        vmin=0,
        vmax=len(np.unique(labels)) - 1,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if centroids is not None:
        centroid_vals = np.array(list(centroids.values()))
        centroid_keys = np.array(list(centroids.keys()))
        ax.scatter(
            centroid_vals[:, 0],
            centroid_vals[:, 1],
            c=centroid_keys,
            marker="X",
            s=300,
            cmap=cmap,
            edgecolors="black",
        )

    if test_embeddings is not None:
        ax.scatter(
            test_embeddings[:, 0],
            test_embeddings[:, 1],
            c=test_labels,
            cmap=cmap,
            marker="o",
            s=50,
        )

    if handles is not None:
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    if save and title is not None:
        save_path = os.path.join("images", title + ".png")
        fig.savefig(save_path, format="png", dpi=300)

    if return_fig:
        return fig
    else:
        plt.show()


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


def create_handles(y: np.ndarray, cmap: Colormap) -> List[mpatches.Patch]:
    """
    Creates a list of legend handles for a given set of labels using a specified colormap.

    Each unique value in y is mapped to a patch colored according to the colormap. This is useful
    for creating legends in visualizations (e.g., t-SNE plots) where the colors correspond to specific labels.

    Args:
        y (Union[np.ndarray, List[Any]]): A 1D array or list of labels (can be numerical or strings).
        cmap (Colormap): A matplotlib colormap object to map label indices to colors.

    Returns:
        List[mpatches.Patch]: A list of matplotlib Patch objects that can be used in a legend.

    Example:
        >>> handles = create_handles(y_train, plt.get_cmap('tab10'))
        >>> plt.legend(handles=handles, title="User ID")
    """
    unique_ids = np.unique(y)
    handles = []
    num_ids = len(unique_ids)
    for i in range(num_ids):
        color = cmap(float(i) / (len(unique_ids) - 1))
        patch = mpatches.Patch(color=color, label=str(unique_ids[i]))
        handles.append(patch)
    return handles


def plot_distance_distribution_on_ax(
    embeddings: np.ndarray,
    participant_ids: np.ndarray,
    ax: plt.Axes,
    distance_type: str = "euclidean",
    bins: int = 30,
):
    """
    Given a set of embeddings (shape=(N, D)) and their participant IDs (shape=(N,)),
    compute pairwise distances (genuine vs. imposter) and plot two histograms
    (genuine in blue, imposter in orange) onto the provided Axes.

    Args:
        embeddings (np.ndarray): Array of shape (N, D).
        participant_ids (np.ndarray): 1-D array of shape (N,) containing integer or string IDs.
        ax (plt.Axes): Matplotlib Axes on which to draw the histograms.
        distance_type (str): "euclidean" or "cosine". Defaults to "euclidean".
        bins (int): Number of bins for each histogram. Defaults to 30.
    """
    N = embeddings.shape[0]

    all_dists = cdist(embeddings, embeddings, metric=distance_type)

    genuine = []
    imposter = []
    for i in range(N):
        for j in range(i + 1, N):
            d = all_dists[i, j]
            if participant_ids[i] == participant_ids[j]:
                genuine.append(d)
            else:
                imposter.append(d)

    ax.hist(
        genuine,
        bins=bins,
        alpha=0.5,
        color="tab:blue",
        label="Genuine",
        density=False,
    )
    ax.hist(
        imposter,
        bins=bins,
        alpha=0.5,
        color="tab:orange",
        label="Imposter",
        density=False,
    )
    ax.set_title(f"{distance_type.capitalize()} Distances")
    ax.set_xlabel(f"{distance_type.capitalize()} Distance")
    ax.set_ylabel("Count")
    ax.legend()


def plot_distance_distribution_return(
    embeddings: np.ndarray,
    participant_ids: np.ndarray,
    distance_type: str = "euclidean",
    bins: int = 30,
    figsize: tuple[int, int] = (6, 4),
) -> plt.Figure:
    """
    Returns a Figure containing histograms of genuine vs. imposter distances.
    """
    fig, ax = plt.subplots(figsize=figsize)
    N = embeddings.shape[0]
    all_dists = cdist(embeddings, embeddings, metric=distance_type)

    genuine, imposter = [], []
    for i in range(N):
        for j in range(i + 1, N):
            (genuine if participant_ids[i] == participant_ids[j] else imposter).append(
                all_dists[i, j]
            )

    ax.hist(genuine, bins=bins, alpha=0.5, color="tab:blue", label="Genuine")
    ax.hist(imposter, bins=bins, alpha=0.5, color="tab:orange", label="Imposter")

    ax.set_title(f"{distance_type.title()} Distances")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_threshold_metrics_return(
    thresholds: np.ndarray,
    fnr_list: np.ndarray,
    fpr_list: np.ndarray,
    acc_list: np.ndarray,
    best_threshold: float,
    best_fnr: float,
    best_fpr: float,
    best_acc: float,
    figsize: tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Returns a two‐panel figure showing FNR/FPR vs threshold and Accuracy vs threshold.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.plot(thresholds, fnr_list, label="FNR", color="tab:blue")
    ax1.plot(thresholds, fpr_list, label="FPR", color="tab:orange")
    ax1.axvline(
        best_threshold, color="white", ls="--", label=f"T*={best_threshold:.3f}"
    )
    ax1.set_title("FNR & FPR vs Threshold")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Error Rate")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(thresholds, acc_list, label="Accuracy", color="tab:green")
    ax2.axvline(
        best_threshold,
        color="white",
        ls="--",
        label=f"Acc={best_acc*100:.1f}%@{best_threshold:.3f}",
    )
    ax2.set_title("Accuracy vs Threshold")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("Threshold Selection")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_threshold_metrics(
    thresholds: np.ndarray,
    fnr_list: np.ndarray,
    fpr_list: np.ndarray,
    acc_list: np.ndarray,
    best_threshold: float,
    best_fnr: float,
    best_fpr: float,
    best_acc: float,
) -> None:
    """
    Produce a two‐panel plot:
      (a) FNR & FPR vs. Threshold
      (b) Accuracy vs. Threshold
    and mark the best_threshold as a vertical line.

    Args:
        thresholds (np.ndarray): shape = (num_thresholds,)
        fnr_list (np.ndarray): shape = (num_thresholds,)
        fpr_list (np.ndarray): shape = (num_thresholds,)
        acc_list (np.ndarray): shape = (num_thresholds,)
        best_threshold (float): threshold which maximizes accuracy
        best_fnr (float): FNR at best_threshold
        best_fpr (float): FPR at best_threshold
        best_acc (float): Accuracy at best_threshold
    """
    plt.figure(figsize=(12, 5))

    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(thresholds, fnr_list, label="False‐Reject Rate (FNR)", color="tab:blue")
    ax1.plot(thresholds, fpr_list, label="False‐Accept Rate (FPR)", color="tab:orange")
    ax1.axvline(
        best_threshold,
        color="white",
        linestyle="--",
        linewidth=1.2,
        label=f"Chosen T = {best_threshold:.3f}",
    )
    ax1.set_xlabel("Distance Threshold")
    ax1.set_ylabel("Error Rate")
    ax1.set_title("FNR & FPR vs. Threshold")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(thresholds, acc_list, label="Overall Accuracy", color="tab:green")
    ax2.axvline(
        best_threshold,
        color="white",
        linestyle="--",
        linewidth=1.2,
        label=f"max Acc = {best_acc*100:.1f}% at T = {best_threshold:.3f}",
    )
    ax2.set_xlabel("Distance Threshold")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Verification Accuracy vs. Threshold")
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)

    plt.suptitle("Threshold Selection: FNR, FPR and Overall Accuracy")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_f1_vs_threshold_return(
    thresholds: np.ndarray,
    f1_list: np.ndarray,
    best_threshold: float,
    best_f1: float,
    figsize: tuple[int, int] = (8, 5),
) -> plt.Figure:
    """
    Returns a figure of F1‐score vs threshold, marking the best point.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(thresholds, f1_list, color="tab:purple", label="F1")
    ax.axvline(
        best_threshold,
        color="white",
        ls="--",
        label=f"F1={best_f1:.3f}@{best_threshold:.3f}",
    )
    ax.set_title("F1 vs Threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 Score")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_f1_vs_threshold(
    thresholds: np.ndarray, f1_list: np.ndarray, best_threshold: float, best_f1: float
) -> None:
    """
    Draw F1‐score versus threshold, and mark the chosen best_threshold.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1_list, color="tab:purple", label="F₁ Score")
    plt.axvline(
        best_threshold,
        color="white",
        linestyle="--",
        linewidth=1.2,
        label=f"best T = {best_threshold:.3f}\nmax F₁ = {best_f1:.3f}",
    )

    plt.title("F₁ Score vs. Distance Threshold")
    plt.xlabel("Distance Threshold")
    plt.ylabel("F₁ Score")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_umap(
    embeddings, labels, n_neighbors=15, min_dist=0.1, title="UMAP of EEG Embeddings"
):
    """
    Plots a 2D UMAP visualization of the given embeddings and labels with a legend for the labels.

    Parameters:
    - embeddings (torch.Tensor or np.ndarray): Shape (N, D)
    - labels (torch.Tensor or np.ndarray): Shape (N,)
    - n_neighbors (int): UMAP n_neighbors parameter
    - min_dist (float): UMAP min_dist parameter
    - title (str): Plot title
    """

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42
    )
    embeddings_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        idx = labels == label
        plt.scatter(
            embeddings_2d[idx, 0], embeddings_2d[idx, 1], s=10, label=f"Label {label}"
        )

    plt.title(title)
    plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()


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
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=10, c="blue")
    plt.title("2D Visualization of Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.show()


def plot_embeddings_by_participant(
    embeddings_2d: np.ndarray, participant_ids: np.ndarray
):
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
            s=10,
        )

    plt.title("2D Visualization of Embeddings by Participant")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
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
            s=10,
        )

    plt.title("UMAP Visualization of Embeddings by Participant")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
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
            s=10,
        )

    plt.title("PCA Visualization of Test Participants")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
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
            s=10,
        )

    plt.title("t-SNE Visualization of Test Participants")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
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
            s=10,
        )

    plt.title("LDA Visualization of Test Participants")
    plt.xlabel("LDA Dimension 1")
    plt.ylabel("LDA Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
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
    plt.xlabel("Time")
    plt.title("Sample signal from first channel")
    plt.ylabel("Amplitude")
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
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
    plt.title(f"Sample signal from all channels")
    plt.legend(["channel 1", "channel 2", "channel 3", "channel 4"])
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
    for signal in eeg_df["epoch"].iloc:
        plt.plot(signal[0], alpha=0.5)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
    plt.title(f"First channel signal from all participants")
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
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (dB/Hz)")
    plt.title("Power Spectral Density (PSD) for all channels for sample")
    plt.legend(["channel 1", "channel 2", "channel 3", "channel 4"])
    plt.show()
