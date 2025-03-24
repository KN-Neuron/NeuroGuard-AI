import matplotlib.pyplot as plt
import numpy as np
import os

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


def plot_tsne(embeddings,
              cmap,
              labels,
              handles=None,
              figsize=(10, 8),
              alpha=1.0,
              title=None,
              xlabel=None,
              ylabel=None,
              centroids=None,
              test_embeddings=None,
              test_labels=None,
              save=False):
    plt.figure(figsize=(figsize[0], figsize[1]))
    scatter_train = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap=cmap, alpha=alpha)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if centroids is not None:
        centroid_vals = np.array(list(centroids.values()))
        centroid_keys = np.array(list(centroids.keys()))
        scatter_centroid = plt.scatter(centroid_vals[:, 0], centroid_vals[:, 1], c=centroid_keys, marker="X", s=300,
                               cmap=cmap, edgecolors="black")

    if test_embeddings is not None:
        scatter_test = plt.scatter(test_embeddings[:, 0],
                                   test_embeddings[:, 1],
                                   c=test_labels,
                                   cmap=cmap,
                                   marker="o",
                                   s=50)
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if save:
        save_path = os.path.join("images", title + ".png")
        plt.savefig(save_path, format="png", dpi=300)

    plt.show()

