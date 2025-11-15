import torch
from torch.utils.data import DataLoader
from typing import Any, List, Tuple

def extract_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader[Any],
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    embeddings_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            anchor = batch[0].to(device)
            labels = batch[3]  # assume label is always at index 3

            embeddings = model(anchor)  # shape: (B, embedding_dim)
            embeddings_list.append(embeddings.cpu())
            labels_list.append(torch.tensor(labels))  # convert list to tensor if needed

    all_embeddings = torch.cat(embeddings_list, dim=0)  # shape: (N, D)
    all_labels = torch.cat(labels_list, dim=0)          # shape: (N,)
    return all_embeddings, all_labels

def extract_embeddings_online(
    model: torch.nn.Module,
    dataloader: DataLoader[Any],
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    embeddings_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            embeddings = model(inputs)  # shape: (batch_size, embedding_dim)
            embeddings_list.append(embeddings.cpu())
            labels_list.append(labels.cpu())

    all_embeddings = torch.cat(embeddings_list, dim=0)  # shape: (N, embedding_dim)
    all_labels = torch.cat(labels_list, dim=0)          # shape: (N,)
    return all_embeddings, all_labels

