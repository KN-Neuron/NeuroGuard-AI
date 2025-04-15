import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from eeg_lib.data.datasets import WelchDatasetKolory
from eeg_lib.models.similarity.apt_model import APT

val_dataset = WelchDatasetKolory("datasets", persons_left=27, reversed_persons=False)

# Load model

model = APT(
    freq_dim=100,
    electorode_dim=4,
    num_classes=27,
    loss_relation=.3,
)
model.load_state_dict(
    torch.load("apt_model_prime.pth", weights_only=True, map_location=torch.device("cuda"))
)
model.to(torch.device("cuda"))
model.eval()

# Load data
X, Y = list(zip(*val_dataset))
X = torch.row_stack(X).reshape((-1, 4, 100))
Y = torch.row_stack(Y)

labels = torch.argmax(Y, dim=1)
embeddings = model(X.to(model.device), return_embeddings=True)

with torch.no_grad():
    all_positive_distances = torch.empty(size=(0,))
    all_negative_distances = torch.empty(size=(0,))

    for label in range(3):
        positive_embeddings = embeddings[labels == label]
        negative_embeddings = embeddings[labels != label]

        for embedding in positive_embeddings:
            # distances = 1 - F.cosine_similarity(positive_embeddings, embedding)
            distances = torch.norm(positive_embeddings - embedding, p=2, dim=1)
            all_positive_distances = torch.cat((all_positive_distances, distances.cpu()))
            
        for embedding in negative_embeddings:
            # distances = 1 - F.cosine_similarity(positive_embeddings, embedding)
            distances = torch.norm(positive_embeddings - embedding, p=2, dim=1)
            all_negative_distances = torch.cat((all_negative_distances, distances.cpu()))

plt.hist(all_positive_distances, 20, color="green", alpha=.7)
plt.hist(all_negative_distances, 20, color="red", alpha=.7)
plt.show()
