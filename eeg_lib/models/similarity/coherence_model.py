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
            nn.Tanh(),
        )

        self.classifier_layer = nn.Linear(240, num_classes)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.embedding_loss = nn.TripletMarginWithDistanceLoss(
        #     margin=0.5, distance_function=lambda x, y: 1 - F.cosine_similarity(x, y)
        # )
        self.embedding_loss = nn.HingeEmbeddingLoss()
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
        writer = SummaryWriter()
        
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            running_hinge_loss = 0.0
            running_class_loss = 0.0

            for batch in train_loader:
                batch_loss = 0

                # Embedding Loss
                distances, targets, embeddings, labels = self.get_pairs(batch)
                hinge_loss = self.embedding_loss(distances, targets)
                batch_loss += hinge_loss

                # Classification Loss
                classification = self.classifier_layer(embeddings)
                classification_loss = self.classification_loss(classification, labels)
                batch_loss += classification_loss

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                running_loss += batch_loss.item()
                running_hinge_loss += hinge_loss.item()
                running_class_loss += classification_loss.item()
                
            if val_loader is not None:
                val_loss, val_hinge_loss, val_class_loss = self.evaluate(val_loader)

            avg_loss = running_loss / len(train_loader)
            avg_hinge_loss = running_hinge_loss / len(train_loader)
            avg_class_loss = running_class_loss / len(train_loader)

            writer.add_scalar('Loss/train', avg_loss, epoch)
            writer.add_scalar('Hinge_Loss/train', avg_hinge_loss, epoch)
            writer.add_scalar('Classification_Loss/train', avg_class_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Hinge_Loss/val', val_hinge_loss, epoch)
            writer.add_scalar('Classification_Loss/val', val_class_loss, epoch)
            
            print(epoch, avg_loss, val_loss)

    def get_pairs(self, batch) -> torch.Tensor:
        X, labels = batch
        X = X.to(self.device)
        labels = labels.to(self.device)

        embeddings = self.embedding_net(X)
        return_distances = torch.empty(size=(len(X),), dtype=torch.float32)
        return_targets = torch.empty(size=(len(X),), dtype=torch.float32)

        for ind, (anchor, label) in enumerate(zip(embeddings, labels)):
            if torch.rand((1,)) > 0.5:
                positive_idx = torch.where(labels == label)[0]
                distances = torch.norm(embeddings[positive_idx] - anchor, p=2, dim=1)

                return_distances[ind] = torch.max(distances)
                return_targets[ind] = 1
            else:
                negative_idx = torch.where(labels != label)[0]
                distances = torch.norm(embeddings[negative_idx] - anchor, p=2, dim=1)

                return_distances[ind] = torch.min(distances)
                return_targets[ind] = -1

        return (
            return_distances.to(self.device),
            return_targets.to(self.device),
            embeddings,
            labels,
        )

    def evaluate(self, data_loader: DataLoader) -> tuple[float, float, float]:
        self.eval()
        running_loss = 0.0
        running_hinge_loss = 0.0
        running_class_loss = 0.0

        with torch.no_grad():
            for batch in data_loader:
                distances, targets, embeddings, labels = self.get_pairs(batch)

                # Calculate losses
                hinge_loss = self.embedding_loss(distances, targets)
                classification = self.classifier_layer(embeddings)
                classification_loss = self.classification_loss(classification, labels)
                batch_loss = hinge_loss + classification_loss

                running_loss += batch_loss.item()
                running_hinge_loss += hinge_loss.item()
                running_class_loss += classification_loss.item()

        avg_loss = running_loss / len(data_loader)
        avg_hinge_loss = running_hinge_loss / len(data_loader)
        avg_class_loss = running_class_loss / len(data_loader)

        return avg_loss, avg_hinge_loss, avg_class_loss

    def visualize_embeddings(self, dataloader) -> None:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.discriminant_analysis import \
            LinearDiscriminantAnalysis as LDA
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

        X = PCA(n_components=2).fit_transform(embeddings, targets)

        for y in torch.unique(targets):
            plt.scatter(X[:, 0][targets == y], X[:, 1][targets == y])
        plt.show()
