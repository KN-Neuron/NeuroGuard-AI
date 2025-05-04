import numpy as np
import torch
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset
from scipy import signal

from eeg_lib.data.data_loader.custom_data_loader import get_coh_features
from eeg_lib.data.data_loader.EEGDataExtractor import EEGDataExtractor


def _compute_or_load_coherence(directory: str) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        X = torch.load(f"{directory}/CohKolory_data.pt")
        y = torch.load(f"{directory}/CohKolory_labels.pt")

    except FileNotFoundError:
        extractor = EEGDataExtractor(
            data_dir=f"{directory}/Kolory",
            hfreq=55,
            resample_freq=100,
            tmax=10,
        )
        eeg_df, _ = extractor.extract_dataframe()

        X = np.array(
            [*eeg_df["epoch"][~eeg_df["label"].isin(["grey", "break"])].to_numpy()]
        )
        X = get_coh_features(X.transpose(0, 2, 1))
        X = torch.tensor(X, dtype=torch.float32)
        torch.save(X, f"{directory}/CohKolory_data.pt")

        y = LabelEncoder().fit_transform(
            eeg_df["participant_id"][~eeg_df["label"].isin(["grey", "break"])]
        )
        y = torch.tensor(
            OneHotEncoder(sparse_output=False).fit_transform(y.reshape((-1, 1)))
        )
        torch.save(y, f"{directory}/CohKolory_labels.pt")

    return X, y


def _compute_or_load_welch(directory: str) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        psds = torch.load(f"{directory}/WelchKolory_data.pt")
        y = torch.load(f"{directory}/WelchKolory_labels.pt")

    except FileNotFoundError:
        fs = 251
        window_size = fs
        overlap = window_size // 2

        extractor = EEGDataExtractor(
            data_dir=f"{directory}/Kolory",
            resample_freq=100,
            tmax=5,
        )
        eeg_df, _ = extractor.extract_dataframe()

        X = np.array(
            [*eeg_df["epoch"][~eeg_df["label"].isin(["grey", "break"])].to_numpy()]
        )
        X = torch.tensor(X, dtype=torch.float32)

        psds = torch.empty(size=(X.shape[0], X.shape[1], 100), dtype=torch.float32)
        for t_i, trial in enumerate(X):
            for e_i, electrode in enumerate(trial):
                freqs, psd = signal.welch(
                    electrode,
                    fs=fs,
                    window="hann",
                    nperseg=window_size,
                    noverlap=overlap,
                )

                psd = torch.from_numpy(psd[0:100])
                psds[t_i, e_i] = torch.log10(psd)
            psds[t_i] = psds[t_i] = (psds[t_i] - psds[t_i].mean()) / psds[t_i].std()

        torch.save(psds, f"{directory}/WelchKolory_data.pt")

        y = LabelEncoder().fit_transform(
            eeg_df["participant_id"][~eeg_df["label"].isin(["grey", "break"])]
        )
        y = torch.tensor(
            OneHotEncoder(sparse_output=False).fit_transform(y.reshape((-1, 1)))
        )
        torch.save(y, f"{directory}/WelchKolory_labels.pt")

    return psds, y


class CohDatasetKolory(Dataset):
    def __init__(
        self,
        directory: str,
        persons_left: int | None = None,
        reversed_persons: bool = False,
    ):
        super().__init__()

        self.X, self.y = _compute_or_load_coherence(directory)

        if not reversed_persons:
            inds = self.y[:, -persons_left:].sum(dim=1) == 0
            self.y = self.y[inds, :-persons_left]
        else:
            inds = self.y[:, -persons_left:].sum(dim=1) != 0
            self.y = self.y[inds, -persons_left:]

        self.X = self.X[inds]

        self.num_classes = len(self.y[0])
        self.input_size = self.X[0].shape[0]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class WelchDatasetKolory(Dataset):
    def __init__(
        self,
        directory: str,
        persons_left: int | None = None,
        reversed_persons: bool = False,
    ):
        super().__init__()

        self.X, self.y = _compute_or_load_welch(directory)

        if not reversed_persons:
            inds = self.y[:, -persons_left:].sum(dim=1) == 0
            self.y = self.y[inds, :-persons_left]
        else:
            inds = self.y[:, -persons_left:].sum(dim=1) != 0
            self.y = self.y[inds, -persons_left:]

        self.X = self.X[inds]

        self.num_classes = len(self.y[0])
        self.freq_dim = self.X.shape[2]
        self.electorode_dim = self.X.shape[1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


if __name__ == "__main__":
    X, y = _compute_or_load_welch(
        "/home/vanilla/Studia/neuron/rnns/artificial-intelligence/datasets"
    )

    import matplotlib.pyplot as plt

    for x in X:
        for p in x:
            plt.plot(p)
        plt.show()
