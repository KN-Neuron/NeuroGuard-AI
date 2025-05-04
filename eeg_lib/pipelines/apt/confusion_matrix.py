import torch
import matplotlib.pyplot as plt

from eeg_lib.data.datasets import WelchDatasetKolory
from eeg_lib.models.similarity.apt_model import APT

val_dataset = WelchDatasetKolory("datasets", persons_left=3, reversed_persons=True)

# Load model
model = APT(
    freq_dim=val_dataset.freq_dim,
    electrode_dim=val_dataset.electorode_dim,
    num_classes=27,
    loss_relation=0.9,
)
model.load_state_dict(
    torch.load(
        "apt_model_prime.pth", weights_only=True, map_location=torch.device("cuda")
    )
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
# Assumes: embeddings (N, D) are L2-normalized
# labels: (N,) with values 0, 1, 2

k = 5
max_authorization_samples = 10
best_accuracy = 0
best_results = None
best_threshold = None

for t in range(0, 8):
    threshold = 0.2 + t * 0.05
    results = torch.zeros((3, 2, 2))

    for label in range(3):
        all_class_embeddings = embeddings[labels == label]
        all_other_embeddings = embeddings[labels != label]

        if all_class_embeddings.shape[0] > max_authorization_samples:            
            perm = torch.randperm(all_class_embeddings.shape[0])
            auth_indices = perm[:min(10, len(perm))]
            query_indices = perm[min(10, len(perm)):]  # disjoint

            auth_embeddings = all_class_embeddings[auth_indices]
            query_embeddings = all_class_embeddings[query_indices]

        # === Positive probes ===
        sims_pos = torch.matmul(query_embeddings, auth_embeddings.T)  # (Q, 10)
        topk_sim_pos, _ = sims_pos.topk(k, dim=1)
        pos_pass = (topk_sim_pos >= (1 - threshold)).any(dim=1)

        results[label][0][0] = pos_pass.sum().item()  # TP
        results[label][0][1] = (~pos_pass).sum().item()  # FN
        results[label][0] /= query_embeddings.shape[0]

        # === Negative probes ===
        sims_neg = torch.matmul(all_other_embeddings, auth_embeddings.T)  # (N, 10)
        topk_sim_neg, _ = sims_neg.topk(k, dim=1)
        neg_pass = (topk_sim_neg >= (1 - threshold)).any(dim=1)

        results[label][1][0] = neg_pass.sum().item()  # FP
        results[label][1][1] = (~neg_pass).sum().item()  # TN
        results[label][1] /= all_other_embeddings.shape[0]

    # Compute accuracy
    tp = results[:, 0, 0].sum()
    tn = results[:, 1, 1].sum()
    fp = results[:, 1, 0].sum()
    fn = results[:, 0, 1].sum()
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_results = results.clone()
        best_threshold = threshold

print(f"Best threshold: {round(best_threshold, 2)} (Accuracy: {best_accuracy:.4f})")
print(best_results)
