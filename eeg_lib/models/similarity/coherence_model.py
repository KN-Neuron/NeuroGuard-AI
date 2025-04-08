import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

def contrastive_loss(x1, x2, y, margin=1.0):
    dist = torch.norm(x1 - x2, p=2, dim=1)
    loss = (1 - y) * dist.pow(2) + y * torch.clamp(margin - dist, min=0).pow(2)
    return loss.mean()


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
            nn.Tanh()
        )

        self.classifier_layer = nn.Linear(240, num_classes)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.embedding_loss = nn.TripletMarginWithDistanceLoss(
        #     margin=0.5, distance_function=lambda x, y: 1 - F.cosine_similarity(x, y)
        # )
        self.embedding_loss = nn.HingeEmbeddingLoss(margin=.1)
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

            # for anchor, positive, negative in train_loader:
            #     anchor = anchor.to(self.device)
            #     positive = positive.to(self.device)
            #     negative = negative.to(self.device)

            #     anchor_emb = self.embedding_net(anchor)
            #     positive_emb = self.embedding_net(positive)
            #     negative_emb = self.embedding_net(negative)

            #     # Compute losses
            #     triplet_loss = self.embedding_loss(
            #         anchor_emb, positive_emb, negative_emb
            #     )
            #     running_loss += triplet_loss

            #     self.optimizer.zero_grad()
            #     triplet_loss.backward()
            #     self.optimizer.step()

            for anchor, pair, target, label in train_loader:
                batch_loss = 0
                anchor = anchor.to(self.device)
                pair = pair.to(self.device)
                target = target.to(self.device)
                label = label.to(self.device)

                # Embedding Loss
                anchor_emb = self.embedding_net(anchor)
                pair_emb = self.embedding_net(pair)
                distances = torch.norm(anchor_emb - pair_emb, p=2, dim=1)
                # hinge_loss = self.embedding_loss(anchor_emb, pair_emb, target)
                hinge_loss = self.embedding_loss(distances, target)
                batch_loss += hinge_loss
                
                # Classification Loss
                classification = self.classifier_layer(anchor_emb)
                classification_loss = self.classification_loss(classification, label)
                batch_loss += 0.4*classification_loss

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                
                running_loss += batch_loss.item()

            print(epoch, running_loss / len(train_loader))

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
            for inputs, _, _, labels in dataloader:
                emb = self(inputs.to(self.device), return_embeddings=True)

                embeddings.append(emb.cpu())
                targets.append(labels)

        embeddings = torch.cat(embeddings)
        targets = torch.argmax(torch.cat(targets), dim=1)

        X = LDA(n_components=2).fit_transform(embeddings, targets)

        for y in torch.unique(targets):
            plt.scatter(X[:, 0][targets == y], X[:, 1][targets == y])
        plt.show()
