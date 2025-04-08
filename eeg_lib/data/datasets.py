import numpy as np
import torch
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset, random_split

from .data_loader.custom_data_loader import get_coh_features
from .data_loader.EEGDataExtractor import EEGDataExtractor


def _compute_or_load_coherence(directory: str) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        X = torch.load(f"{directory}/CohKolory_data.pt")
        y = torch.load(f"{directory}/CohKolory_labels.pt")

    except FileNotFoundError:
        extractor = EEGDataExtractor(
            data_dir="../../artificial-intelligence/data/kolory/Kolory",
            hfreq=55,
            resample_freq=100,
            tmax=10,
        )
        eeg_df, _ = extractor.extract_dataframe()

        X = np.array([*eeg_df["epoch"].to_numpy()])
        X = get_coh_features(X.transpose(0, 2, 1))
        X = torch.tensor(X, dtype=torch.float32)
        torch.save(X, f"{directory}/CohKolory_data.pt")

        y = LabelEncoder().fit_transform(eeg_df["participant_id"])
        y = torch.tensor(
            OneHotEncoder(sparse_output=False).fit_transform(y.reshape((-1, 1)))
        )
        torch.save(y, f"{directory}/CohKolory_labels.pt")

    return X, y


class CohDatasetKolory(Dataset):
    def __init__(self, directory: str):
        super().__init__()

        self.X, self.y = _compute_or_load_coherence(directory)

        self.num_classes = len(self.y[0])
        self.input_size = self.X[0].shape[0]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class CohDatasetKolory_Triplets(Dataset):
    def __init__(self, directory: str):
        super().__init__()

        self.X, self.y = _compute_or_load_coherence(directory)

        self.num_classes = len(self.y[0])
        self.input_size = self.X[0].shape[0]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        anchor = self.X[index]
        label = self.y[index]

        positive_idx = torch.where(self.y == label)[0]
        negative_idx = torch.where(self.y != label)[0]
        
        positive = self.X[positive_idx[torch.randint(len(positive_idx), (1,)).item()]]
        negative = self.X[negative_idx[torch.randint(len(negative_idx), (1,)).item()]]
        
        return anchor, positive, negative

class CohDatasetKolory_Pairs(Dataset):
    def __init__(self, directory: str, model):
        super().__init__()
        self.model = model
        
        self.X, self.y = _compute_or_load_coherence(directory)

        self.num_classes = len(self.y[0])
        self.input_size = self.X[0].shape[0]
        

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        anchor = self.X[index]
        label = self.y[index]

        if torch.rand((1,)) > .5:
            positive_idx = torch.where(self.y == label)[0]
            positive = self.X[positive_idx[torch.randint(len(positive_idx), (1,)).item()]]
            return anchor, positive, 1, label
        else:
            negative_idx = torch.where(self.y != label)[0]            
            negative = self.X[negative_idx[torch.randint(len(negative_idx), (1,)).item()]]            
            return anchor, negative, -1, label
