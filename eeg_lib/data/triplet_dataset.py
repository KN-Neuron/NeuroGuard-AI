from torch.utils.data import Dataset
import random
import torch

class TripletEEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.label_to_indices = self._create_label_dict()

    def _create_label_dict(self):
        label_to_indices = {}
        for idx, label in enumerate(self.labels):
            label = label.item()
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        return label_to_indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        anchor = self.data[index]
        anchor_label = self.labels[index].item()

        positive_idx = index
        while positive_idx == index:
            positive_idx = random.choice(self.label_to_indices[anchor_label])
        positive = self.data[positive_idx]

        negative_label = random.choice([l for l in self.label_to_indices if l != anchor_label])
        negative_idx = random.choice(self.label_to_indices[negative_label])
        negative = self.data[negative_idx]

        return anchor, positive, negative, anchor_label


class HardTripletEEGDataset(Dataset):
    def __init__(self, data, labels, model, device='cpu'):
        self.data = data
        self.labels = labels
        self.label_to_indices = self._create_label_dict()
        self.model = model.to(device)
        self.device = device

        if len(self.label_to_indices) < 2:
            raise ValueError("Dataset must contain at least 2 classes for triplet sampling")
        self.embeddings = self._compute_embeddings()

    def _create_label_dict(self):
        label_to_indices = {}
        for idx, label in enumerate(self.labels):
            label = label.item()
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        return label_to_indices

    def _compute_embeddings(self):
        self.model.eval()
        with torch.no_grad():
            embeddings = []
            for x in self.data:
                x = x.unsqueeze(0).to(self.device)
                embedding = self.model(x)
                embeddings.append(embedding.cpu())
            return torch.stack(embeddings)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        anchor = self.data[index]
        anchor_label = self.labels[index].item()
        anchor_embedding = self.embeddings[index]

        positive_indices = [i for i in self.label_to_indices[anchor_label] if i != index]

        if not positive_indices:
            positive = anchor
        else:
            pos_embeddings = self.embeddings[positive_indices]
            dists = torch.norm(pos_embeddings - anchor_embedding, dim=-1)
            hardest_positive_idx = positive_indices[torch.argmax(dists)]
            positive = self.data[hardest_positive_idx]

        negative_indices = []
        for lbl in self.label_to_indices:
            if lbl != anchor_label:
                negative_indices.extend(self.label_to_indices[lbl])

        if not negative_indices:
            available_labels = [lbl for lbl in self.label_to_indices if lbl != anchor_label]
            if not available_labels:
                raise RuntimeError("No negative samples available for triplet loss")
            chosen_label = random.choice(available_labels)
            negative_indices = self.label_to_indices[chosen_label]

        neg_embeddings = self.embeddings[negative_indices]
        dists = torch.norm(neg_embeddings - anchor_embedding, dim=-1)
        hardest_negative_idx = negative_indices[torch.argmin(dists)]
        negative = self.data[hardest_negative_idx]

        return anchor, positive, negative, anchor_label

class SimpleEEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]



