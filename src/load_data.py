import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from pandas import read_csv
from scipy import signal


class EEGDataExtractor:
    def __init__(
        self,
        data_dir: str,
        lfreq: float = 1,
        hfreq: float = 100,
        notch_filter: list = [50],
        baseline: Optional[Tuple] = None,
        tmin: float = 0,
        tmax: float = 3,
    ):
        """
        Parameters:
            data_dir (str): Directory with .fif files.
            lfreq (float): Low cutoff frequency for bandpass filtering.
            hfreq (float): High cutoff frequency for bandpass filtering.
            notch_filter (list): Frequencies for notch filtering (e.g., to remove 50Hz line noise).
            baseline (tuple): Baseline correction period.
            tmin (float): Start time (in seconds) relative to the event.
            tmax (float): End time (in seconds) relative to the event.
        """
        self.data_dir = data_dir
        self.lfreq = lfreq
        self.hfreq = hfreq
        self.notch_filter = notch_filter
        self.baseline = baseline
        self.tmin = tmin
        self.tmax = tmax

    def _read_from_dir(self):
        """Returns a list of .fif files in the data directory."""
        return [f for f in os.listdir(self.data_dir) if f.endswith(".fif")]

    def _load_eeg(self):
        """
        Loads each .fif file, applies filtering, converts units,
        extracts events and epochs, and maps event codes to labels.
        """
        files = self._read_from_dir()
        eeg_and_events = []
        participants = []
        for file in files:
            participant_id = os.path.splitext(file)[0]
            file_path = os.path.join(self.data_dir, file)
            eeg = mne.io.read_raw_fif(file_path, preload=True)
            eeg.pick_types(eeg=True, stim=False, eog=False, exclude="bads")
            # Convert units (from µV to V if needed)
            eeg.apply_function(lambda x: x * 10**-6)
            eeg.filter(l_freq=self.lfreq, h_freq=self.hfreq)
            eeg.notch_filter(self.notch_filter)
            events, event_id = mne.events_from_annotations(eeg)
            if not event_id:
                print(f"No events found in file {file}")
                continue
            # reverse mapping: integer code -> label name (e.g., 1 -> 'red')
            id_to_label = {v: k for k, v in event_id.items()}
            epochs = mne.Epochs(
                raw=eeg,
                events=events,
                event_id=event_id,
                tmin=self.tmin,
                tmax=self.tmax,
                baseline=self.baseline,
                preload=True,
            )
            numeric_labels = epochs.events[:, -1]
            labels = [id_to_label.get(l, "unknown") for l in numeric_labels]
            eeg_and_events.append(
                {"participant_id": participant_id, "epochs": epochs, "labels": labels}
            )
            participants.append({"participant_id": participant_id, "file": file})
        return eeg_and_events, participants

    def extract_dataframe(self) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Iterates over each participant's data, extracts each epoch as a numpy array,
        and returns a DataFrame with columns: participant_id, epoch, label.
        Also returns a list of participants with metadata.
        """
        eeg_and_events, participants = self._load_eeg()
        data = []
        for item in eeg_and_events:
            participant_id = item["participant_id"]
            epochs = item["epochs"]
            labels = item["labels"]
            # Retrieve all epochs as a 3D np array: (n_epochs, n_channels, n_times)
            epoch_data = epochs.get_data()
            for i, label in enumerate(labels):
                epoch_array = epoch_data[i]
                data.append(
                    {
                        "participant_id": participant_id,
                        "epoch": epoch_array,
                        "label": label,
                    }
                )
        df = pd.DataFrame(data)
        return df, participants


def downsample(X: np.ndarray, fs: float, f: int) -> np.ndarray:
    ratio = fs // f
    X_ = X[: X.shape[0] - X.shape[0] % ratio]
    return np.mean(X_.reshape(X_.shape[0] // ratio, ratio, -1), axis=1)


def butter_bandpass_filter(
    X: np.ndarray, fs: int = 500, lowcut: int = 1, highcut: int = 55, order: int = 4
) -> None:
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = signal.butter(order, [low, high], btype="band")

    return np.apply_along_axis(lambda x: signal.filtfilt(b, a, x), axis=0, arr=X)


def epochs_p300(
    X: np.ndarray, y: np.ndarray, time_range: tuple[int, int] = (-200, 1000)
) -> np.ndarray:
    """
    This function assumes the data is 100Hz
    - `y` represents array with each row being a pair (*trigger, target*)
    - `time_range` is passed in ms
    """
    epochs = []
    for ind, (_, target) in enumerate(y):
        if target:
            epochs.append(X[ind + time_range[0] // 10 : ind + time_range[1] // 10])
    return np.array(epochs)


def epochs_coh(X: np.ndarray, time_per_epoch: float) -> np.ndarray:
    epoch_size = int(time_per_epoch * 100)

    trimmed_X = X[: -(X.shape[0] % epoch_size)]
    return trimmed_X.reshape((-1, epoch_size, X.shape[1]))


def get_avg_features(X: np.ndarray) -> np.ndarray:
    features = []
    for i in range(len(X)):
        features.append(
            np.mean((tmp := X[i, 30:]).reshape(tmp.shape[0] // 10, 10, -1), axis=1)
        )

    return np.array(features)


def get_coh_features(X: np.ndarray) -> np.ndarray:
    fs = 100
    window_size = fs
    overlap = window_size // 2

    num_epochs, _, num_channels = X.shape
    coherence_features = np.zeros(
        (num_epochs, (num_channels * (num_channels - 1) // 2) * 40)
    )

    for epoch_idx, epoch_data in enumerate(X):
        feature_vector = []
        for i, channel_i in enumerate(epoch_data.T):
            for j, channel_j in enumerate(epoch_data.T[i + 1 :], start=i + 1):
                freqs, coh_values = signal.coherence(
                    channel_i,
                    channel_j,
                    fs=fs,
                    window="hann",
                    nperseg=window_size,
                    noverlap=overlap,
                )
                coh_values: np.ndarray = coh_values[
                    1:41
                ]  # Take first 40 frequency bins
                normalized_coh_values = (
                    coh_values - coh_values.mean()
                ) / coh_values.std()  # Normalize
                feature_vector.extend(normalized_coh_values)

        coherence_features[epoch_idx, :] = feature_vector

    return coherence_features


def extract_avg_features(
    data: np.ndarray, subject_index: int, flatten=True
) -> tuple[np.ndarray, np.ndarray]:
    # y refers to whether or not the shown picture is target or not
    # it does NOT represent labels

    # Data preprocessing
    data[:, :-2] = butter_bandpass_filter(data[:, :-2], fs=512)
    data = downsample(data, 512, 100)
    X, y = data[1:, :-2], data[:, -2:].astype(np.bool)

    # Feature extraction
    X = epochs_p300(X, y)
    X = get_avg_features(X)
    y = np.full(shape=len(X), fill_value=subject_index, dtype=np.uint16)

    if flatten:
        return X.reshape((X.shape[0], -1)), y
    return X, y


def extract_coh_features(
    data: np.ndarray,
    subject_index: int,
    flatten: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    # Data preprocessing
    X = butter_bandpass_filter(data, fs=512, highcut=50)
    X = downsample(X, 512, 100)

    # Feature extraction
    X = epochs_coh(X, 10.0)
    X = get_coh_features(X)
    y = np.full(shape=len(X), fill_value=subject_index, dtype=np.uint16)

    if flatten:
        return X, y
    return X.reshape((X.shape[0], 40, -1))


if __name__ == "__main__":
    data = read_csv("data/subject_03_session_01.csv", header=None).to_numpy()
    X1, y1 = extract_coh_features(data, 0)
    data = read_csv("data/subject_03_session_02.csv", header=None).to_numpy()
    X2, y2 = extract_coh_features(data, 1)
    data = read_csv("data/subject_03_session_03.csv", header=None).to_numpy()
    X3, y3 = extract_coh_features(data, 2)

    X = np.vstack((X1, X2, X3))
    y = np.hstack((y1, y2, y3))

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    print(X.shape)
    print(y.shape)

    reg = LogisticRegression()
    reg.fit(X, y)

    print(accuracy_score(y, reg.predict(X)))
