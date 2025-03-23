import numpy as np
from poetry.console.commands import self


class SimilarityCentroidsVerifier(object):
    def __init__(self,embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
        self.centroids = self.compute_centroids()

    def compute_centroids(self):
        centroids = {}
        for label in np.unique(self.labels):
            centroids[label] = self.embeddings[self.labels == label].mean(axis=0)
        return centroids

    def euclidean_distance(self,a, b):
        return np.linalg.norm(a-b)
    def compute_similarity(self, embedding, label):
        centroid = self.centroids[label]
        distance = self.euclidean_distance(embedding, centroid)
        return distance

    def get_avg_distance(self):
        avg_distances = {}
        unique_labels = np.unique(self.labels)
        for label in unique_labels:
            mask = self.labels == label
            embeddings_label = self.embeddings[mask]
            distances = np.linalg.norm(embeddings_label - self.centroids[label], axis=1)
            avg_distances[label] = np.mean(distances)
        return avg_distances
