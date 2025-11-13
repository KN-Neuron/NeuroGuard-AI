import numpy as np
from poetry.console.commands import self
import torch
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Union, List, Tuple, Iterable, Any


class SimilarityCentroidsVerifier(object):
    """A verifier that classifies embeddings based on their Euclidean distance
    to class centroids. Supports both batch and online updates of centroids.

    Attributes
    ----------
    alpha : float
        Momentum factor for online centroid updates (0 < alpha < 1).
    centroids : dict
        Maps class labels to their centroid vectors.
    labels : np.ndarray or list[int]
        The labels of the embeddings used to compute the true centroids.
    embeddings : np.ndarray or torch.Tensor
        The embeddings used to compute the true centroids.

    """
    def __init__(self, alpha: float = 0.9) -> None:
        """Initialize the verifier.

        Parameters
        ----------
        alpha : float, optional
            Weight for exponential smoothing when updating centroids online, by default 0.9.
        """
        self.alpha = 0.9
        self.centroids = {}

    def compute_true_centroids(self, labels: Union[np.ndarray, List[int]],
                               embeddings: Union[np.ndarray, torch.Tensor]) -> Dict[int, np.ndarray]:
        """Compute the exact centroids for each class from scratch.

        This method overwrites any existing centroids.

        Parameters
        ----------
        labels : np.ndarray or list[int]
            Array of class labels for each embedding.
        embeddings : np.ndarray or torch.Tensor
            Array of embeddings of shape (N, D), where N is the number of samples
            and D is the embedding dimension.

        Returns
        -------
        dict[int, np.ndarray]
            A dictionary mapping each unique label to its computed centroid vector.
        """
        self.labels = labels
        self.embeddings = embeddings
        centroids = {}
        for label in np.unique(labels):
            centroids[label] = embeddings[labels == label].mean(axis=0)
        self.centroids = centroids
        return centroids

    def update_centroids(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        """Perform an online (momentum-based) update of centroids given a new batch.

        If a label from the new batch is not already in `self.centroids`, it is
        added. Otherwise, the existing centroid is updated using exponential moving average.

        Parameters
        ----------
        embeddings : torch.Tensor
            A batch of embeddings, with shape (B, D), where B is the batch size.
        labels : torch.Tensor
            The corresponding class labels for the embeddings, with shape (B,).
        """
        labels_np = labels.cpu().numpy()
        unique_labels = np.unique(labels_np)
        embeddings_cpu = embeddings.cpu()
        for label in unique_labels:
            mask = (labels.cpu() == label)
            new_centroid = embeddings_cpu[mask].mean(dim=0).detach().numpy()
            if label in self.centroids:
                self.centroids[label] = self.alpha * self.centroids[label] + (1 - self.alpha) * new_centroid
            else:
                self.centroids[label] = new_centroid

    def get_centroids(self) -> Dict[int, np.ndarray]:
        """Return the current centroids.

        Returns
        -------
        dict[int, np.ndarray]
            A dictionary mapping from label to centroid vector.
        """
        return self.centroids

    def euclidean_distance(self, a: Union[np.ndarray, torch.Tensor],
                           b: Union[np.ndarray, torch.Tensor]) -> float:
        """Compute Euclidean distance between two vectors.

        Parameters
        ----------
        a : np.ndarray or torch.Tensor
            The first vector.
        b : np.ndarray or torch.Tensor
            The second vector.

        Returns
        -------
        float
            The Euclidean distance between vectors `a` and `b`.
        """
        return np.linalg.norm(a - b)

    def compute_similarity(self, embedding: Union[np.ndarray, torch.Tensor], label: int) -> float:
        """Compute distance from an embedding to the centroid of a specific class.

        Parameters
        ----------
        embedding : np.ndarray or torch.Tensor
            A single embedding vector.
        label : int
            The class label whose centroid to compare against.

        Returns
        -------
        float
            The Euclidean distance from the embedding to the specified class centroid.
        """
        embedding_np = embedding.cpu().detach().numpy() if isinstance(embedding, torch.Tensor) else embedding
        centroid = self.centroids[label]
        distance = self.euclidean_distance(embedding_np, centroid)
        return distance

    def classify_embedding(self, embedding: Union[np.ndarray, torch.Tensor]) -> Tuple[Any, float]:
        """Assign a label to a single embedding by nearest-centroid.

        Parameters
        ----------
        embedding : np.ndarray or torch.Tensor
            The embedding vector to classify.

        Returns
        -------
        tuple[any, float]
            A tuple containing the predicted class label and the
            distance to that class's centroid.
        """
        min_distance = None
        min_label = None
        for label in self.centroids.keys():
            distance = self.compute_similarity(embedding, label)
            if min_distance is None or distance < min_distance:
                min_distance = distance
                min_label = label
        return min_label, min_distance

    def classify_batch(self, embeddings: Iterable[Union[np.ndarray, torch.Tensor]]) -> List[Tuple[Any, float]]:
        """Classify a batch of embeddings.

        Parameters
        ----------
        embeddings : Iterable[np.ndarray or torch.Tensor]
            A list or other iterable of embedding vectors to classify.

        Returns
        -------
        list[tuple[any, float]]
            A list of tuples, where each tuple contains the predicted label and
            the distance to the centroid for a corresponding input embedding.
        """
        predictions = []
        for emb in embeddings:
            predictions.append(self.classify_embedding(emb))
        return predictions

    def get_avg_distance(self) -> Dict[int, float]:
        """Compute the average distance of each class's samples to its centroid.

        This method relies on the `labels` and `embeddings` stored from the last
        call to `compute_true_centroids`.

        Returns
        -------
        dict[int, float]
            A dictionary mapping each label to the average distance of its
            member embeddings to the class centroid.
        """
        avg_distances = {}
        unique_labels = np.unique(self.labels)
        for label in unique_labels:
            mask = self.labels == label
            embeddings_label = self.embeddings[mask]
            distances = np.linalg.norm(embeddings_label - self.centroids[label], axis=1)
            avg_distances[label] = np.mean(distances)
        return avg_distances


def get_accuracy(embd: np.ndarray, test_embd: np.ndarray,
                 y_train_encoded: np.ndarray, y_test_encoded: np.ndarray) -> Tuple[float, float]:
    """Calculates training and testing accuracy using a nearest centroid classifier.

    The function first normalizes the training and testing embeddings to the
    range [0, 1]. It then computes centroids from the training data and uses
    them to classify both training and testing sets, reporting the accuracy for each.

    Parameters
    ----------
    embd : np.ndarray
        Training embeddings of shape (n_samples, n_features).
    test_embd : np.ndarray
        Testing embeddings of shape (n_samples_test, n_features).
    y_train_encoded : np.ndarray
        Encoded labels for the training embeddings.
    y_test_encoded : np.ndarray
        Encoded labels for the testing embeddings.

    Returns
    -------
    tuple[float, float]
        A tuple containing the training accuracy and testing accuracy.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_normalized = scaler.fit_transform(embd)
    test_normalized = scaler.transform(test_embd)
    centroid_verifier_actual = SimilarityCentroidsVerifier()

    centroid_verifier_actual.compute_true_centroids(y_train_encoded, train_normalized)
    train_acc = 0
    for i in range(len(train_normalized)):
        true_label = y_train_encoded[i]
        predicted_label, _ = centroid_verifier_actual.classify_embedding(train_normalized[i])
        if predicted_label == true_label:
            train_acc += 1
    train_acc = train_acc / len(train_normalized)

    centroid_verifier_actual_test = SimilarityCentroidsVerifier()
    centroid_verifier_actual_test.compute_true_centroids(y_test_encoded, test_normalized)
    test_acc = 0
    for i in range(len(test_normalized)):
        true_label = y_test_encoded[i]
        predicted_label, _ = centroid_verifier_actual_test.classify_embedding(test_normalized[i])
        if predicted_label == true_label:
            test_acc += 1
    test_acc = test_acc / len(test_normalized)

    return train_acc, test_acc