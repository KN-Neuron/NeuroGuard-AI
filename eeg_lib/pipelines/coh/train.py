import torch
from torch.utils.data import DataLoader

from eeg_lib.data.datasets import CohDatasetKolory
from eeg_lib.models.similarity.coherence_model import BasicModel

train_dataset = CohDatasetKolory("datasets", persons_left=27, reversed_persons=True)
val_dataset = CohDatasetKolory("datasets", persons_left=27, reversed_persons=False)

train_loader = DataLoader(train_dataset, batch_size=512*2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)

model = BasicModel(input_size=train_dataset.input_size, num_classes=train_dataset.num_classes, loss_relation=.8)

model.load_state_dict(torch.load("coh_model.pth", weights_only=True, map_location=torch.device("cuda")))
model.to(torch.device("cuda"))

model.train_model(train_loader, val_loader=val_loader, lr=.001, epochs=20)

model.visualize_embeddings(train_loader)
model.visualize_embeddings(val_loader)

torch.save(model.state_dict(), "coh_model_prime.pth")