import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as mpatches
from typing import Optional, List, Union, Tuple, Any
from matplotlib.colors import Colormap, ListedColormap
from scipy.spatial.distance import cdist

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
    return_fig: bool = True
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

    scatter_train = ax.scatter(embeddings[:, 0], embeddings[:, 1],
                               c=labels, cmap=cmap, alpha=alpha,
                               vmin=0, vmax=len(np.unique(labels)) - 1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if centroids is not None:
        centroid_vals = np.array(list(centroids.values()))
        centroid_keys = np.array(list(centroids.keys()))
        ax.scatter(centroid_vals[:, 0], centroid_vals[:, 1],
                   c=centroid_keys, marker="X", s=300, cmap=cmap,
                   edgecolors="black")

    if test_embeddings is not None:
        ax.scatter(test_embeddings[:, 0], test_embeddings[:, 1],
                   c=test_labels, cmap=cmap, marker="o", s=50)

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
    # Compute the full N×N distance matrix once:
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
    figsize: tuple[int,int]=(6,4),
) -> plt.Figure:
    """
    Returns a Figure containing histograms of genuine vs. imposter distances.
    """
    fig, ax = plt.subplots(figsize=figsize)
    N = embeddings.shape[0]
    all_dists = cdist(embeddings, embeddings, metric=distance_type)

    genuine, imposter = [], []
    for i in range(N):
        for j in range(i+1, N):
            (genuine if participant_ids[i]==participant_ids[j] else imposter).append(all_dists[i,j])

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
    figsize: tuple[int,int]=(12,5),
) -> plt.Figure:
    """
    Returns a two‐panel figure showing FNR/FPR vs threshold and Accuracy vs threshold.
    """
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=figsize)

    ax1.plot(thresholds, fnr_list, label="FNR", color="tab:blue")
    ax1.plot(thresholds, fpr_list, label="FPR", color="tab:orange")
    ax1.axvline(best_threshold, color="white", ls="--", label=f"T*={best_threshold:.3f}")
    ax1.set_title("FNR & FPR vs Threshold")
    ax1.set_xlabel("Threshold"); ax1.set_ylabel("Error Rate")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(thresholds, acc_list, label="Accuracy", color="tab:green")
    ax2.axvline(best_threshold, color="white", ls="--",
                label=f"Acc={best_acc*100:.1f}%@{best_threshold:.3f}")
    ax2.set_title("Accuracy vs Threshold")
    ax2.set_xlabel("Threshold"); ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("Threshold Selection")
    plt.tight_layout(rect=[0,0,1,0.95])
    return fig


def plot_threshold_metrics(
    thresholds: np.ndarray,
    fnr_list: np.ndarray,
    fpr_list: np.ndarray,
    acc_list: np.ndarray,
    best_threshold: float,
    best_fnr: float,
    best_fpr: float,
    best_acc: float
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

    # (a) FNR & FPR vs. Threshold
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(thresholds, fnr_list, label="False‐Reject Rate (FNR)", color="tab:blue")
    ax1.plot(thresholds, fpr_list, label="False‐Accept Rate (FPR)", color="tab:orange")
    ax1.axvline(
        best_threshold, color="white", linestyle="--", linewidth=1.2,
        label=f"Chosen T = {best_threshold:.3f}"
    )
    ax1.set_xlabel("Distance Threshold")
    ax1.set_ylabel("Error Rate")
    ax1.set_title("FNR & FPR vs. Threshold")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    # (b) Accuracy vs. Threshold
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(thresholds, acc_list, label="Overall Accuracy", color="tab:green")
    ax2.axvline(
        best_threshold, color="white", linestyle="--", linewidth=1.2,
        label=f"max Acc = {best_acc*100:.1f}% at T = {best_threshold:.3f}"
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
    figsize: tuple[int,int]=(8,5),
) -> plt.Figure:
    """
    Returns a figure of F1‐score vs threshold, marking the best point.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(thresholds, f1_list, color="tab:purple", label="F1")
    ax.axvline(best_threshold, color="white", ls="--",
               label=f"F1={best_f1:.3f}@{best_threshold:.3f}")
    ax.set_title("F1 vs Threshold")
    ax.set_xlabel("Threshold"); ax.set_ylabel("F1 Score")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig



def plot_f1_vs_threshold(
    thresholds: np.ndarray,
    f1_list: np.ndarray,
    best_threshold: float,
    best_f1: float
) -> None:
    """
    Draw F1‐score versus threshold, and mark the chosen best_threshold.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1_list, color="tab:purple", label="F₁ Score")
    plt.axvline(
        best_threshold,
        color="white", linestyle="--", linewidth=1.2,
        label=f"best T = {best_threshold:.3f}\nmax F₁ = {best_f1:.3f}"
    )
    plt.title("F₁ Score vs. Distance Threshold")
    plt.xlabel("Distance Threshold")
    plt.ylabel("F₁ Score")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()