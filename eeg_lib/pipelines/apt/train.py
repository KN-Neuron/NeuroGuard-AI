import torch
from torch.utils.data import DataLoader

from eeg_lib.data.datasets import WelchDatasetKolory
from eeg_lib.models.similarity.apt_model import APT

train_dataset = WelchDatasetKolory("datasets", persons_left=3)
val_dataset = WelchDatasetKolory("datasets", persons_left=3, reversed_persons=True)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)

model = APT(
    freq_dim=train_dataset.freq_dim,
    electrode_dim=train_dataset.electorode_dim,
    num_classes=train_dataset.num_classes,
    loss_relation=0.9,
)

model.load_state_dict(torch.load("apt_model_prime.pth", weights_only=True, map_location=torch.device("cuda")))
model.to(torch.device("cuda"))
model.visualize_embeddings(val_loader)

# model.train_model(train_loader, val_loader=val_loader, lr=0.001, epochs=20)

# model.visualize_embeddings(train_loader)
# model.visualize_embeddings(val_loader)

# torch.save(model.state_dict(), "apt_model_prime.pth")
