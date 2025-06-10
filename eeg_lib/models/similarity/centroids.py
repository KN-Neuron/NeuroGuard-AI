import numpy as np
from poetry.console.commands import self
import torch

class SimilarityCentroidsVerifier(object):
    """
    A verifier that classifies embeddings based on their Euclidean distance
    to class centroids. Supports both batch and online updates of centroids.

    Attributes:
        alpha (float): Momentum factor for online centroid updates (0 < alpha < 1).
        centroids (dict): Maps class labels to their centroid vectors.
    """
    def __init__(self, alpha=0.9):
        """
        Initialize the verifier.

        Args:
            alpha (float): Weight for exponential smoothing when updating centroids online.
        """
        self.alpha = 0.9
        self.centroids = {}

    def compute_true_centroids(self, labels, embeddings):
        """
        Compute the exact centroids for each class from scratch.

        Args:
            labels (array-like[int]): Array of class labels for each embedding.
            embeddings (ndarray,float or Tensor): Array of embeddings of shape (N, D).

        Returns:
            dict: Mapping from each unique label to its centroid vector.
        """
        self.labels = labels
        self.embeddings = embeddings
        centroids = {}
        for label in np.unique(labels):
            centroids[label] = embeddings[labels == label].mean(axis=0)
        self.centroids = centroids
        return centroids

    def update_centroids(self, embeddings, labels):
        """
        Perform an online (momentum-based) update of centroids given a new batch.

        Args:
            embeddings (Tensor): Batch of embeddings, shape (B, D).
            labels (Tensor): Corresponding class labels, shape (B,).
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

    def get_centroids(self):
        """
        Return the current centroids.

        Returns:
            dict: Mapping from label to centroid vector.
        """
        return self.centroids

    def euclidean_distance(self, a, b):
        """
        Compute Euclidean distance between two vectors.

        Args:
            a (ndarray or Tensor): First vector.
            b (ndarray or Tensor): Second vector.

        Returns:
            float: Euclidean distance.
        """
        return np.linalg.norm(a - b)

    def compute_similarity(self, embedding, label):
        """
        Compute distance from an embedding to the centroid of a specific class.

        Args:
            embedding (ndarray or Tensor): Single embedding vector.
            label (int): Class label whose centroid to compare.

        Returns:
            float: Euclidean distance to the class centroid.
        """
        embedding_np = embedding.cpu().detach().numpy() if isinstance(embedding, torch.Tensor) else embedding
        centroid = self.centroids[label]
        distance = self.euclidean_distance(embedding_np, centroid)
        return distance

    def classify_embedding(self, embedding):
        """
        Assign a label to a single embedding by nearest-centroid.

        Args:
            embedding (ndarray or Tensor): Embedding to classify.

        Returns:
            tuple: (predicted_label, distance_to_centroid)
        """
        min_distance = None
        min_label = None
        for label in self.centroids.keys():
            distance = self.compute_similarity(embedding, label)
            if min_distance is None or distance < min_distance:
                min_distance = distance
                min_label = label
        return min_label, min_distance

    def classify_batch(self, embeddings):
        """
        Classify a batch of embeddings.

        Args:
            embeddings (Iterable[ndarray or Tensor]): List or array of embeddings.

        Returns:
            list of tuples: Each tuple is (predicted_label, distance).
        """
        predictions = []
        for emb in embeddings:
            predictions.append(self.classify_embedding(emb))
        return predictions

    def get_avg_distance(self):
        """
        Compute average distance of each class's samples to its centroid.

        Returns:
            dict: Mapping from label to average distance.
        """
        avg_distances = {}
        unique_labels = np.unique(self.labels)
        for label in unique_labels:
            mask = self.labels == label
            embeddings_label = self.embeddings[mask]
            distances = np.linalg.norm(embeddings_label - self.centroids[label], axis=1)
            avg_distances[label] = np.mean(distances)
        return avg_distances