import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns

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


def plot_loss_and_accuracy(history):
    """Plot training curves with proper metrics for authentication systems"""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Triplet Margin Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Authentication Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_embeddings(embeddings_data, model_name="EEGNet", perplexity=30, color_based=False):
    """
    Visualize embeddings using t-SNE.

    Parameters:
    -----------
    embeddings_data : dict
        Dictionary containing embeddings, participant_ids, and labels
        from extract_embeddings function
    model_name : str
        Name of the model for plot title
    perplexity : int
        Perplexity parameter for t-SNE (typical values: 5-50)
    color_based : bool
        Whether to color points by color label instead of participant ID
    """
    # Apply t-SNE to reduce embeddings to 2D for visualization
    embeddings = embeddings_data['embeddings']
    participant_ids = embeddings_data['participant_ids']
    labels = embeddings_data['labels']

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'tsne_1': embeddings_2d[:, 0],
        'tsne_2': embeddings_2d[:, 1],
        'participant_id': participant_ids,
        'label': labels
    })

    # Create visualization
    plt.figure(figsize=(12, 10))

    if color_based:
        # Color by label (red, green, blue, etc.)
        sns.scatterplot(
            x='tsne_1', y='tsne_2',
            hue='label',
            data=df,
            palette='viridis',
            s=50, alpha=0.7
        )
        plt.title(f'{model_name} Embeddings by Color')
    else:
        # Color by participant ID
        sns.scatterplot(
            x='tsne_1', y='tsne_2',
            hue='participant_id',
            data=df,
            palette='tab20',
            s=50, alpha=0.7
        )
        plt.title(f'{model_name} Embeddings by Participant')

    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    return embeddings_2d