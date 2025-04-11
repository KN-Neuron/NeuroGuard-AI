import torch
from torch.utils.data import DataLoader

from eeg_lib.data.datasets import CohDatasetKolory
from eeg_lib.models.similarity.coherence_model import BasicModel

train_dataset = CohDatasetKolory("datasets", persons_left=3)
val_dataset = CohDatasetKolory("datasets", persons_left=3, reversed_persons=True)

train_loader = DataLoader(train_dataset, batch_size=256*2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)

model = BasicModel(input_size=train_dataset.input_size, num_classes=train_dataset.num_classes, loss_relation=.5)

# model.load_state_dict(torch.load("coh_model4.pth", weights_only=True, map_location=torch.device("cuda")))
# model.to(torch.device("cuda"))

model.train_model(train_loader, val_loader=val_loader, lr=.001, epochs=10)

model.visualize_embeddings(val_loader)

torch.save(model.state_dict(), "coh_model.pth")