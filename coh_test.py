import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset, random_split

from eeg_lib.data.data_loader.custom_data_loader import get_coh_features
from eeg_lib.data.data_loader.EEGDataExtractor import EEGDataExtractor


# Define the LSTM model
class COHModel(nn.Module):
    def __init__(self, input_size=240, num_classes=32):
        super(COHModel, self).__init__()

        self.layers = nn.ModuleList(
            [
                nn.Linear(input_size, 240),
                nn.Linear(240, 240),
                nn.Linear(240, 240),
                nn.Linear(240, 240),
                nn.Linear(240, num_classes),
            ]
        )

    def forward(self, x, return_embeddings=False):
        for layer in self.layers[:-1]:
            x = F.tanh(layer(x))

        return self.layers[-1](x) if not return_embeddings else x


class CustomDatasetKolory(Dataset):
    def __init__(self):
        super().__init__()
        extractor = EEGDataExtractor(
            data_dir="../../artificial-intelligence/data/kolory/Kolory",
            hfreq=55,
            resample_freq=100,
            tmax=10,
        )
        eeg_df, _ = extractor.extract_dataframe()

        self.X = np.array([*eeg_df["epoch"].to_numpy()])
        self.X = self.X.transpose(0, 2, 1)
        self.X = get_coh_features(self.X)

        self.X = torch.tensor(self.X, dtype=torch.float32)

        self.y = LabelEncoder().fit_transform(eeg_df["participant_id"])
        self.y = torch.tensor(
            OneHotEncoder(sparse_output=False).fit_transform(self.y.reshape((-1, 1)))
        )

        self.num_classes = len(self.y[0])
        self.input_size = self.X[0].shape[0]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = CustomDatasetKolory()

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

print(dataset.input_size)
print(dataset.num_classes)
train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


model = COHModel(input_size=dataset.input_size, num_classes=dataset.num_classes).to(
    device
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

num_epochs = 12
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_y = batch_y.to(device)
            outputs = model(batch_X.to(device))

            _, predicted = torch.max(outputs, 1)
            _, labels = torch.max(batch_y, 1)

            correct += (predicted == labels).sum().item()
            total += batch_y.size(0)

    epoch_loss = running_loss / len(train_loader)
    print(
        f"Epoch [{epoch+1:2}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {correct / total:.2f}"
    )

torch.save(model.state_dict(), "coh_model.pth")


def get_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            batch_embeddings = model(inputs, return_embeddings=True)
            embeddings.append(batch_embeddings.cpu())
            labels.append(torch.argmax(targets, 1))

    return torch.cat(embeddings).numpy(), torch.cat(labels).numpy()


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


def visualize_embeddings(embeddings, labels):
    # Reduce to 2D with PCA
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap="tab20", alpha=0.6
    )

    # Add legend and labels
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("TSNE of Model Embeddings")
    plt.xlabel("TSNE Component 1")
    plt.ylabel("TSNE Component 2")
    plt.colorbar(label="Class")
    plt.show()


val_embeddings, val_labels = get_embeddings(model, val_loader, device)

from sklearn.preprocessing import StandardScaler

embeddings = StandardScaler().fit_transform(val_embeddings)

visualize_embeddings(val_embeddings, val_labels)
