import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np



def add_gaussian_noise(eeg, std=0.01):
    noise = np.random.normal(0, std, eeg.shape)
    return eeg + noise



class TDNNDataset(Dataset):
    def __init__(self, data, labels, augmentation=False, std=0.01):

        self.data = data
        self.labels = labels
        self.augmentation = augmentation
        self.std = std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        eeg = self.data[index]
        if self.augmentation:
            eeg = add_gaussian_noise(eeg, self.std)
        x = torch.tensor(eeg, dtype=torch.float32)
        y = torch.tensor(self.labels[index], dtype=torch.long)
        return x, y