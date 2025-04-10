import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class BasicModel(nn.Module):
    def __init__(self, input_size=240, num_classes=32, pair_type_per_class: int = 10):
        super(BasicModel, self).__init__()
        self.pair_type_per_class = pair_type_per_class

        self.embedding_net = nn.Sequential(
            nn.Linear(input_size, 240),
            nn.ELU(),
            nn.Linear(240, 960),
            nn.ELU(),
            nn.Linear(960, 960),
            nn.ELU(),
            nn.Linear(960, 960),
            nn.ELU(),
            nn.Linear(960, 960),
            nn.ELU(),
            nn.Linear(960, 960),
            nn.ELU(),
            nn.Linear(960, 240),
        )

        self.classifier_layer = nn.Sequential(
            nn.Linear(240, num_classes),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1 - torch.sum(x * y, dim=1), swap=True
        )
        self.classification_loss = nn.CrossEntropyLoss()
        self.optimizer = None

    def forward(self, x, return_embeddings=False):
        embeddings = self.embedding_net(x)
        embeddings = F.normalize(x, p=2, dim=1)

        return embeddings if return_embeddings else self.classifier_layer(embeddings)

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 50,
        lr: float = 0.001,
    ) -> None:
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.to(self.device)
        writer = SummaryWriter()

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            running_triplet_loss = 0.0
            running_class_loss = 0.0

            for batch in train_loader:
                batch_loss = 0

                # Embedding Loss
                anchors, positives, negatives, labels = self.get_triplets(batch)
                triplet_loss = self.embedding_loss(anchors, positives, negatives)
                batch_loss += triplet_loss

                # Classification Loss
                classification = self.classifier_layer(anchors)
                classification_loss = self.classification_loss(classification, labels)
                batch_loss += classification_loss

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                running_loss += batch_loss.item()
                running_triplet_loss += triplet_loss.item()
                running_class_loss += classification_loss.item()

            if val_loader is not None:
                val_loss, val_hinge_loss, val_class_loss = self.evaluate(val_loader)

            avg_loss = running_loss / len(train_loader)
            avg_hinge_loss = running_triplet_loss / len(train_loader)
            avg_class_loss = running_class_loss / len(train_loader)

            writer.add_scalar("Loss/train", avg_loss, epoch)
            writer.add_scalar("Triplet_Loss/train", avg_hinge_loss, epoch)
            writer.add_scalar("Classification_Loss/train", avg_class_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Triplet_Loss/val", val_hinge_loss, epoch)
            writer.add_scalar("Classification_Loss/val", val_class_loss, epoch)

            print(epoch, avg_loss, val_loss)

    def get_triplets(self, batch) -> torch.Tensor:
        X, labels = batch
        X = X.to(self.device)
        labels = labels.to(self.device)

        embeddings = self.embedding_net(X)
        return_positives = torch.empty_like(embeddings, device=self.device)
        return_negatives = torch.empty_like(embeddings, device=self.device)

        for ind, (anchor, label) in enumerate(zip(embeddings, labels)):
            positive_idx = torch.where(labels == label)[0]
            negative_idx = torch.where(labels != label)[0]

            # Positive
            distances = -torch.sum(embeddings[positive_idx] * anchor, dim=1)
            return_positives[ind] = embeddings[positive_idx][torch.argmax(distances)]

            # Negative
            distances = -torch.sum(embeddings[negative_idx] * anchor, dim=1)
            return_negatives[ind] = embeddings[negative_idx][torch.argmin(distances)]

        return (embeddings, return_positives, return_negatives, labels)

    def evaluate(self, data_loader: DataLoader) -> tuple[float, float, float]:
        self.eval()
        running_loss = 0.0
        running_triplet_loss = 0.0
        running_class_loss = 0.0

        with torch.no_grad():
            for batch in data_loader:
                anchors, positives, negatives, labels = self.get_triplets(batch)

                # Calculate losses
                triplet_loss = self.embedding_loss(anchors, positives, negatives)
                classification = self.classifier_layer(anchors)
                classification_loss = self.classification_loss(classification, labels)
                batch_loss = triplet_loss + classification_loss

                running_loss += batch_loss.item()
                running_triplet_loss += triplet_loss.item()
                running_class_loss += classification_loss.item()

        avg_loss = running_loss / len(data_loader)
        avg_hinge_loss = running_triplet_loss / len(data_loader)
        avg_class_loss = running_class_loss / len(data_loader)

        return avg_loss, avg_hinge_loss, avg_class_loss

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

        X = LDA(n_components=2).fit_transform(embeddings, targets)

        for y in torch.unique(targets):
            plt.scatter(X[:, 0][targets == y], X[:, 1][targets == y])
        plt.show()
