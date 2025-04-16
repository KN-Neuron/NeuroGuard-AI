import numpy as np
from poetry.console.commands import self
import torch

class SimilarityCentroidsVerifier(object):
    def __init__(self, alpha=0.9):
        # self.embeddings = embeddings
        # self.labels = labels
        # self.centroids = self.compute_centroids()
        self.alpha = 0.9
        self.centroids = {}

    def compute_true_centroids(self, labels, embeddings):
        self.labels = labels
        self.embeddings = embeddings
        centroids = {}
        for label in np.unique(labels):
            centroids[label] = embeddings[labels == label].mean(axis=0)
        self.centroids = centroids
        return centroids

    def update_centroids(self, embeddings, labels):
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
        return self.centroids

    def euclidean_distance(self, a, b):
        return np.linalg.norm(a - b)

    def compute_similarity(self, embedding, label):
        embedding_np = embedding.cpu().detach().numpy() if isinstance(embedding, torch.Tensor) else embedding
        centroid = self.centroids[label]
        distance = self.euclidean_distance(embedding_np, centroid)
        return distance

    def classify_embedding(self, embedding):
        min_distance = None
        min_label = None
        for label in self.centroids.keys():
            distance = self.compute_similarity(embedding, label)
            if min_distance is None or distance < min_distance:
                min_distance = distance
                min_label = label
        return min_label, min_distance

    def classify_batch(self, embeddings):
        predictions = []
        for emb in embeddings:
            predictions.append(self.classify_embedding(emb))
        return predictions

    def get_avg_distance(self):
        avg_distances = {}
        unique_labels = np.unique(self.labels)
        for label in unique_labels:
            mask = self.labels == label
            embeddings_label = self.embeddings[mask]
            distances = np.linalg.norm(embeddings_label - self.centroids[label], axis=1)
            avg_distances[label] = np.mean(distances)
        return avg_distances