import torch

def extract_embeddings(model, dataloader, device):
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            anchor = batch[0].to(device)
            labels = batch[3]  # assume label is always at index 3

            embeddings = model(anchor)  # shape: (B, embedding_dim)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(torch.tensor(labels))  # convert list to tensor if needed

    all_embeddings = torch.cat(all_embeddings, dim=0)  # shape: (N, D)
    all_labels = torch.cat(all_labels, dim=0)          # shape: (N,)
    return all_embeddings, all_labels

def extract_embeddings_online(model, dataloader, device):
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            embeddings = model(inputs)  # shape: (batch_size, embedding_dim)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)  # shape: (N, embedding_dim)
    all_labels = torch.cat(all_labels, dim=0)          # shape: (N,)
    return all_embeddings, all_labels

