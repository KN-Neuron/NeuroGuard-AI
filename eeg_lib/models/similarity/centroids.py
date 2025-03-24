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

    def classify_embedding(self, embedding):
        min_distance = None
        min_label = None
        for label in np.unique(self.labels):
            distance = self.compute_similarity(embedding, label)
            if min_distance is None or distance < min_distance:
                min_distance = distance
                min_label = label
        return min_label, min_distance

    def get_avg_distance(self):
        avg_distances = {}
        unique_labels = np.unique(self.labels)
        for label in unique_labels:
            mask = self.labels == label
            embeddings_label = self.embeddings[mask]
            distances = np.linalg.norm(embeddings_label - self.centroids[label], axis=1)
            avg_distances[label] = np.mean(distances)
        return avg_distances
