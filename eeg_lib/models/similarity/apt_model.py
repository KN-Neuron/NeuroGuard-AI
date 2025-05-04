import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class APT(nn.Module):
    def __init__(
        self,
        freq_dim=40,
        electrode_dim=4,
        nhead=2,
        num_layers=2,
        dim_feedforward=1024,
        num_classes=30,
        loss_relation=0.7,
    ):
        super().__init__()
        self.loss_relation = loss_relation

        self.electrode_dim = electrode_dim
        self.freq_dim = freq_dim

        self.electrode_embed = nn.Parameter(torch.randn(1, electrode_dim, freq_dim))
        self.freq_embed = nn.Parameter(torch.randn(1, 1, freq_dim))
        self.pos_scale = nn.Parameter(torch.tensor(1.0))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=freq_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            # dropout=0.1,
            batch_first=True,
            activation="gelu",
        )

        self.pre_transformer_layer = nn.Sequential(
            nn.Linear(electrode_dim * freq_dim, electrode_dim * freq_dim),
            nn.Sigmoid(),
            nn.Linear(electrode_dim * freq_dim, electrode_dim * freq_dim),
            nn.Sigmoid(),
            nn.Linear(electrode_dim * freq_dim, electrode_dim * freq_dim),
            nn.BatchNorm1d(electrode_dim * freq_dim),
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.embedding_net = nn.Sequential(
            nn.Linear(freq_dim * electrode_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            # nn.BatchNorm1d(128)
        )

        self.classifier_layer = nn.Sequential(
            nn.Linear(128, num_classes),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1 - torch.sum(x * y, dim=1), margin=0.4
        )
        # self.embedding_loss = nn.TripletMarginLoss(margin=0.4, p=2)
        self.classification_loss = nn.CrossEntropyLoss()
        self.optimizer = None

    def forward(self, x, return_embeddings=False):
        x = self.pre_transformer_layer(x.view(-1, self.electrode_dim * self.freq_dim))
        x = x.view(-1, self.electrode_dim, self.freq_dim)
        x = x + self.pos_scale * self.electrode_embed + self.freq_embed
        partial_embeddings: torch.Tensor = self.transformer(x)
        partial_embeddings = partial_embeddings.flatten(start_dim=1)

        embeddings = self.embedding_net(partial_embeddings)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings if return_embeddings else self.classifier_layer(embeddings)

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 50,
        lr: float = 0.001,
    ) -> None:
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)

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
                batch_loss += self.loss_relation * triplet_loss

                # Classification Loss
                classification = self.classifier_layer(anchors)
                classification_loss = self.classification_loss(classification, labels)
                batch_loss += (1 - self.loss_relation) * classification_loss

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                running_loss += batch_loss.item()
                running_triplet_loss += triplet_loss.item()
                running_class_loss += classification_loss.item()

            if val_loader is not None:
                val_loss, val_hinge_loss, val_class_loss = self.evaluate(val_loader)
                self.visualize_embeddings(val_loader, just_save=True, i=epoch)
            else:
                val_loss, val_hinge_loss, val_class_loss = 0, 0, 0

            avg_loss = running_loss / len(train_loader)
            avg_hinge_loss = running_triplet_loss / len(train_loader)
            avg_class_loss = running_class_loss / len(train_loader)

            writer.add_scalar("Loss/train", avg_loss, epoch)
            writer.add_scalar("Triplet_Loss/train", avg_hinge_loss, epoch)
            writer.add_scalar("Classification_Loss/train", avg_class_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Triplet_Loss/val", val_hinge_loss, epoch)
            writer.add_scalar("Classification_Loss/val", val_class_loss, epoch)

            print(f"{epoch:3}: {avg_loss:5.3f} {val_loss:5.3f}")

    def evaluate(
        self, data_loader: DataLoader, skip_classification: bool = True
    ) -> tuple[float, float, float]:
        self.eval()
        running_loss = 0.0
        running_triplet_loss = 0.0
        running_class_loss = 0.0

        with torch.no_grad():
            for batch in data_loader:
                anchors, positives, negatives, labels = self.get_triplets(batch)

                # Calculate losses
                triplet_loss = self.embedding_loss(anchors, positives, negatives)
                if skip_classification:
                    classification_loss = torch.tensor(
                        (0,), dtype=torch.float64, device=self.device
                    )
                else:
                    classification = self.classifier_layer(anchors)
                    classification_loss = self.classification_loss(
                        classification, labels
                    )
                batch_loss = triplet_loss + classification_loss

                running_loss += batch_loss.item()
                running_triplet_loss += triplet_loss.item()
                running_class_loss += classification_loss.item()

        avg_loss = running_loss / len(data_loader)
        avg_hinge_loss = running_triplet_loss / len(data_loader)
        avg_class_loss = running_class_loss / len(data_loader)

        return avg_loss, avg_hinge_loss, avg_class_loss

    def get_triplets(self, batch) -> torch.Tensor:
        X, labels = batch
        X = X.to(self.device)
        labels = labels.to(self.device)

        embeddings = self(X, return_embeddings=True)
        return_positives = torch.empty_like(embeddings, device=self.device)
        return_negatives = torch.empty_like(embeddings, device=self.device)

        for ind, (anchor, label) in enumerate(zip(embeddings, labels)):
            positive_idx = torch.where(labels == label)[0]
            negative_idx = torch.where(labels != label)[0]

            # Positive
            # distances = torch.norm(embeddings[positive_idx] - anchor, p=2, dim=1)
            distances = -torch.sum(embeddings[positive_idx] * anchor, dim=1)
            if len(distances):
                return_positives[ind] = embeddings[positive_idx][
                    torch.argsort(distances, descending=True)[
                        # 0
                        torch.randint(0, len(distances) // 4, size=(1,))
                    ]
                ]

            # Negative
            # distances = torch.norm(embeddings[negative_idx] - anchor, p=2, dim=1)
            distances = -torch.sum(embeddings[negative_idx] * anchor, dim=1)
            if len(distances):
                return_negatives[ind] = embeddings[negative_idx][
                    torch.argsort(distances)[
                        # 0
                        torch.randint(0, len(distances) // 4, size=(1,))
                    ]
                ]

        return (embeddings, return_positives, return_negatives, labels)

    def visualize_embeddings(self, dataloader, just_save=False, i=0) -> None:
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

        plt.figure()
        X = PCA(n_components=2).fit_transform(embeddings, targets)

        for y in torch.unique(targets):
            plt.scatter(X[:, 0][targets == y], X[:, 1][targets == y], alpha=0.8)

        if not just_save:
            plt.show()
        else:
            plt.savefig(
                f"/home/vanilla/Studia/neuron/rnns/artificial-intelligence/embeddings/emb{i}.png"
            )
        plt.close()


if __name__ == "__main__":
    model = APT(freq_dim=100, dim_feedforward=512, nhead=10, num_layers=6)

    dummy_input = torch.rand(32, 4, 100)

    output = model(dummy_input, return_embeddings=True)

    print(output.shape)
