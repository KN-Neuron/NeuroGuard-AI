import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import torch
import umap.umap_ as umap

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



def plot_tsne(embeddings, labels, perplexity=100, learning_rate=100, title="t-SNE of EEG Embeddings"):
    """
    Plots a 2D t-SNE visualization of the given embeddings and labels with a legend for the labels.

    Parameters:
    - embeddings (torch.Tensor or np.ndarray): Shape (N, D)
    - labels (torch.Tensor or np.ndarray): Shape (N,)
    - perplexity (int): t-SNE perplexity
    - learning_rate (float): t-SNE learning rate
    - title (str): Plot title
    """
    # Convert to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot with legend
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        idx = labels == label
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], s=10, label=f"Label {label}")

    plt.title(title)
    plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.show()

import numpy as np
import torch
import matplotlib.pyplot as plt
import umap

def plot_umap(embeddings, labels, n_neighbors=15, min_dist=0.1, title="UMAP of EEG Embeddings"):
    """
    Plots a 2D UMAP visualization of the given embeddings and labels with a legend for the labels.

    Parameters:
    - embeddings (torch.Tensor or np.ndarray): Shape (N, D)
    - labels (torch.Tensor or np.ndarray): Shape (N,)
    - n_neighbors (int): UMAP n_neighbors parameter
    - min_dist (float): UMAP min_dist parameter
    - title (str): Plot title
    """
    # Convert to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Compute UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)

    # Plot with legend
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        idx = labels == label
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], s=10, label=f"Label {label}")

    plt.title(title)
    plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.show()
