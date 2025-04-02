import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset, random_split

from eeg_lib.data.data_loader.custom_data_loader import get_coh_features
from eeg_lib.data.data_loader.EEGDataExtractor import EEGDataExtractor


class CohDatasetKolory(Dataset):
    def __init__(self):
        super().__init__()
        
        np.load("datasets/KoloryCOH.npy")
        
        extractor = EEGDataExtractor(
            data_dir="../../artificial-intelligence/data/kolory/Kolory",
            hfreq=55,
            resample_freq=100,
            tmax=10,
        )
        eeg_df, _ = extractor.extract_dataframe()

        self.X = np.array([*eeg_df["epoch"].to_numpy()])
        self.X = self.X.transpose(0, 2, 1)
        self.X = get_coh_features(self.X)

        self.X = torch.tensor(self.X, dtype=torch.float32)

        self.y = LabelEncoder().fit_transform(eeg_df["participant_id"])
        self.y = torch.tensor(
            OneHotEncoder(sparse_output=False).fit_transform(self.y.reshape((-1, 1)))
        )

        self.num_classes = len(self.y[0])
        self.input_size = self.X[0].shape[0]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


if __name__ == "__main__":
    d = CohDatasetKolory()