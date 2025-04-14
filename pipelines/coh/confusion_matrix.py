import torch
import matplotlib.pyplot as plt

from eeg_lib.data.datasets import CohDatasetKolory
from eeg_lib.models.similarity.coherence_model import BasicModel

val_dataset = CohDatasetKolory("datasets", persons_left=3, reversed_persons=True)

# Load model
model = BasicModel(input_size=240, num_classes=27)
model.load_state_dict(
    torch.load("coh_model.pth", weights_only=True, map_location=torch.device("cuda"))
)
model.to(torch.device("cuda"))

# Load data
X, Y = list(zip(*val_dataset))
X = torch.row_stack(X)
Y = torch.row_stack(Y)

labels = torch.argmax(Y, dim=1)
embeddings = model(X.to(model.device), return_embeddings=True)
all_positive_distances = torch.empty(size=(0,))
all_negative_distances = torch.empty(size=(0,))

# Verification parameters
n_of_anchors = 10
anchor_threshold = 7

for t in range(0, 10):
    threshold = 1.3 + t*.05
    
    # Result tracking
    results = torch.zeros((3, 2, 2))
    for label in range(3):
        positive_embeddings = embeddings[labels == label]
        negative_embeddings = embeddings[labels != label]

        anchors_idx = torch.randint(0, positive_embeddings.shape[0], (n_of_anchors,))

        for ind, embedding in enumerate(positive_embeddings):
            score = 0
            for ind, anchor in enumerate(positive_embeddings[anchors_idx]):
                dist = torch.norm(anchor - embedding, p=2)
                if dist >= threshold:
                    score += 1

            if score >= anchor_threshold:
                results[label][0][0] += 1
            else:
                results[label][0][1] += 1
                
        results[label][0] /= len(positive_embeddings)
                
        for ind, embedding in enumerate(negative_embeddings):
            score = 0
            for ind, anchor in enumerate(positive_embeddings[anchors_idx]):
                dist = torch.norm(anchor - embedding, p=2)
                if dist >= threshold:
                    score += 1

            if score >= anchor_threshold:
                results[label][1][0] += 1
            else:
                results[label][1][1] += 1

        results[label][1] /= len(negative_embeddings)
        
    print(threshold, results)
