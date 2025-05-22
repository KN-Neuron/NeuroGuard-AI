import torch
import torch.nn as nn
import torch.nn.functional as F


class ProxyNCALoss(nn.Module):
    def __init__(self, num_classes, embedding_dim, scale=None):
        super(ProxyNCALoss, self).__init__()
        self.proxies = nn.Embedding(num_classes, embedding_dim)
        nn.init.kaiming_normal(self.proxies.weight, mode='fan_out')
        if scale is not None:
            self.scale = scale
        else:
            self.scale = nn.Parameter(torch.Tensor([1.0]))

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        proxies = F.normalize(self.proxies.weight, p=2, dim=1)

        similarities = torch.matmul(embeddings, proxies.T)
        logits = self.scale * similarities
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss
