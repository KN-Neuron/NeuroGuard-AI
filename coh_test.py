import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from eeg_lib.data.datasets import CohDatasetKolory, CohDatasetKolory_Triplets, CohDatasetKolory_Pairs
from eeg_lib.models.similarity.coherence_model import BasicModel


dataset = CohDatasetKolory_Pairs("datasets")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    (train_size, val_size),
)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)


model = BasicModel(input_size=dataset.input_size, num_classes=dataset.num_classes, pair_type_per_class=3)

# model.load_state_dict(torch.load("coh_model.pth", weights_only=True, map_location=torch.device("cuda")))
# model.to(torch.device("cuda"))

model.train_model(train_loader, val_loader=val_loader, lr=.001, epochs=10)

model.visualize_embeddings(val_loader)

torch.save(model.state_dict(), "coh_model2.pth")