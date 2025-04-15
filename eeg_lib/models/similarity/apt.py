import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(
        self,
        freq_dim=40,
        electorode_dim=4,
        nhead=10,
        num_layers=6,
        dim_feedforward=512,
        num_classes=30,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=freq_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.embedding_net = nn.Sequential(
            nn.Linear(freq_dim * electorode_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )

        self.classifier_layer = nn.Sequential(
            nn.Linear(128, num_classes),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1 - torch.sum(x * y, dim=1)
        )
        self.classification_loss = nn.CrossEntropyLoss()
        self.optimizer = None

    def forward(self, x, return_embeddings=False):
        partial_embeddings: torch.Tensor = self.transformer(x)
        partial_embeddings = partial_embeddings.flatten(start_dim=1)
        print(partial_embeddings.shape)

        embeddings = self.embedding_net(partial_embeddings)

        return embeddings if return_embeddings else self.classifier_layer(embeddings)

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
            distances = -torch.sum(embeddings[positive_idx] * anchor, dim=1)
            if len(distances):
                return_positives[ind] = embeddings[positive_idx][
                    torch.argmax(distances)
                ]
            else:
                print(negative_idx)
                print(label)

            # Negative
            distances = -torch.sum(embeddings[negative_idx] * anchor, dim=1)
            if len(distances):
                return_negatives[ind] = embeddings[negative_idx][
                    torch.argmin(distances)
                ]
            else:
                print(negative_idx)
                print(label)

        return (embeddings, return_positives, return_negatives, labels)


# Example usage
if __name__ == "__main__":
    model = TransformerModel(freq_dim=40, dim_feedforward=512, nhead=10, num_layers=6)

    dummy_input = torch.rand(32, 4, 40)

    output = model(dummy_input, return_embeddings=True)

    print(output.shape)
