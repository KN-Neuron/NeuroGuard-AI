import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


class ProxyDataset(Dataset):
    def __init__(self, data, labels):

        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        return torch.Tensor(self.data[index]).unsqueeze(0), self.labels[index]