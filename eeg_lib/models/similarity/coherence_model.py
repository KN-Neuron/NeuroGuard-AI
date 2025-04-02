import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from eeg_lib.data.datasets import CohDatasetKolory


class BasicModel(nn.Module):
    def __init__(self, input_size=240, num_classes=32, pair_type_per_class: int = 10):
        super(BasicModel, self).__init__()
        self.pair_type_per_class = pair_type_per_class

        self.embedding_net = nn.Sequential(
            nn.Linear(input_size, 240),
            nn.Sigmoid(),
            nn.Linear(240, 240),
            nn.Sigmoid(),
            nn.Linear(240, 240),
            nn.Sigmoid(),
            nn.Linear(240, 240),
        )

        self.classifier_layer = nn.Linear(240, num_classes)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_loss = nn.TripletMarginLoss(
            margin=0.5,
            p=2,
        )
        self.classification_loss = nn.CrossEntropyLoss()
        self.optimizer = None

    def forward(self, x, return_embeddings=False):
        embeddings = self.embedding_net(x)

        return (
            embeddings
            if return_embeddings
            else self.classifier_layer(F.sigmoid(embeddings))
        )

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 50,
        lr: float = 0.001,
    ) -> None:
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)

        self.to(self.device)

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0

            for batch in train_loader:
                running_loss += self._train_step(batch)

            print(running_loss / len(train_loader))

    def _train_step(self, batch):
        """Single training step"""
        inputs, labels = batch
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        # Generate triplets
        anchors, positives, negatives = self._create_eeg_triplets(inputs, labels)

        # Get embeddings
        anchor_emb = self(anchors, return_embeddings=True)
        positive_emb = self(positives, return_embeddings=True)
        negative_emb = self(negatives, return_embeddings=True)

        loss = self.embedding_loss(anchor_emb, positive_emb, negative_emb)

        # Optional classification loss
        # preds = self(inputs, return_embeddings=False)
        # loss += self.classification_loss(preds, labels)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _create_eeg_triplets(self, inputs: torch.Tensor, labels: torch.Tensor):
        anchors, positives, negatives = [], [], []

        for lbl in torch.unique(labels):
            class_idx = torch.where(labels == lbl)[0]
            other_idx = torch.where(labels != lbl)[0]

            if len(class_idx) < 2 or len(other_idx) < 1:
                continue

            # Create multiple triplets per class
            for _ in range(self.pair_type_per_class):
                # Random anchor and positive
                a, p = torch.randperm(len(class_idx))[:2]
                anchor = inputs[class_idx[a]]
                positive = inputs[class_idx[p]]

                # Hard negative - closest different class sample
                with torch.no_grad():
                    anchor_emb = self(anchor.unsqueeze(0), return_embeddings=True)
                    other_embs = self(inputs[other_idx], return_embeddings=True)
                    dists = torch.cdist(anchor_emb, other_embs).squeeze()
                    hard_neg_idx = torch.argmin(dists)

                anchors.append(anchor)
                positives.append(positive)
                negatives.append(inputs[other_idx[hard_neg_idx]])

        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

    def evaluate(self, dataloader):
        """Validation evaluation"""
        self.eval()
        embeddings, targets = [], []

        with torch.no_grad():
            for inputs, labels in dataloader:
                emb = self(inputs.to(self.device), return_embeddings=True)
                embeddings.append(emb.cpu())
                targets.append(labels)

        embeddings = torch.cat(embeddings)
        targets = torch.cat(targets)

        preds = embeddings.argmax(dim=1) if embeddings.size(1) > 1 else embeddings

        return "xd"

    def visualize_embeddings(self, dataloader) -> None:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        from sklearn.manifold import TSNE

        self.eval()
        embeddings, targets = [], []

        with torch.no_grad():
            for inputs, labels in dataloader:
                emb = self(inputs.to(self.device), return_embeddings=True)

                embeddings.append(emb.cpu())
                targets.append(labels)

        embeddings = torch.cat(embeddings)
        targets = torch.argmax(torch.cat(targets), dim=1)

        X = TSNE(n_components=2).fit_transform(embeddings, targets)

        for y in torch.unique(targets):
            plt.scatter(X[:, 0][targets == y], X[:, 1][targets == y])
        plt.show()
