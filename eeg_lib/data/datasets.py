import numpy as np
import torch
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset, random_split

from .data_loader.custom_data_loader import get_coh_features
from .data_loader.EEGDataExtractor import EEGDataExtractor


class CohDatasetKolory(Dataset):
    def __init__(self, directory: str):
        super().__init__()

        try:
            self.X = torch.load(f"{directory}/CohKolory_data.pt")
            self.y = torch.load(f"{directory}/CohKolory_labels.pt")

        except FileNotFoundError:
            extractor = EEGDataExtractor(
                data_dir="../../artificial-intelligence/data/kolory/Kolory",
                hfreq=55,
                resample_freq=100,
                tmax=10,
            )
            eeg_df, _ = extractor.extract_dataframe()

            self.X = np.array([*eeg_df["epoch"].to_numpy()])
            self.X = get_coh_features(self.X.transpose(0, 2, 1))
            self.X = torch.tensor(self.X, dtype=torch.float32)
            torch.save(self.X, f"{directory}/CohKolory_data.pt")

            self.y = LabelEncoder().fit_transform(eeg_df["participant_id"])
            self.y = torch.tensor(
                OneHotEncoder(sparse_output=False).fit_transform(
                    self.y.reshape((-1, 1))
                )
            )
            torch.save(self.y, f"{directory}/CohKolory_labels.pt")

        self.num_classes = len(self.y[0])
        self.input_size = self.X[0].shape[0]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
