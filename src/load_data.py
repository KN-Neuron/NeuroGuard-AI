from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from scipy import signal


class Preprocessing:
    def __init__(self, path: str, f: int = 5.12):
        self.path = path
        self.f = f

        self.data = read_csv(path, header=None).to_numpy()

        self.X = np.empty(1)
        self.y = np.empty(1)

    def _downsample(self) -> None:
        self.data = self.data[: self.data.shape[0] - self.data.shape[0] % self.f, :]
        self.data = np.mean(
            self.data.reshape(self.data.shape[0] // self.f, self.f, -1), axis=1
        )

    def _get_epochs(self, time_range: tuple[int, int] = (-200, 1000)) -> None:
        epochs = []
        for ind, (_, target) in enumerate(self.y):
            if target:
                epochs.append(
                    self.X[ind + time_range[0] // 10 : ind + time_range[1] // 10]
                )

        self.X = np.array(epochs)

    def _butter_bandpass_filter(
        self, fs: int = 500, lowcut: int = 1, highcut: int = 55, order: int = 4
    ) -> None:
        nyquist = fs / 2
        low = lowcut / nyquist
        high = highcut / nyquist

        b, a = signal.butter(order, [low, high], btype="band")

        self.data[:, 1:-2] = np.apply_along_axis(
            lambda x: signal.filtfilt(b, a, x), axis=0, arr=self.data[:, 1:-2]
        )

    def _get_features(self) -> None:
        features = []
        for i in range(len(self.X)):
            tmp = self.X[i, 30:]
            features.append(np.mean(tmp.reshape(tmp.shape[0] // 10, 10, -1), axis=1))

        return np.array(features)

    def get_subject_data(self, subject_index: int) -> tuple[np.ndarray, np.ndarray]:
        self._butter_bandpass_filter()
        self._downsample()
        self.X, self.y = self.data[1:, :-2], self.data[:, -2:].astype(np.bool)
        self._get_epochs()

        return self._get_features(), np.ones(shape=len(self.X)) * subject_index


def get_data(subjects: list[int], session: int) -> tuple[np.ndarray, np.ndarray]:
    data = list(
        map(
            lambda i: Preprocessing(
                f"data/subject_0{i}_session_0{session}.csv"
            ).get_subject_data(i - min(subjects)),
            subjects,
        )
    )
    X, y = np.vstack([data[i][0] for i in range(len(subjects))]), np.hstack(
        [data[i][1] for i in range(len(subjects))]
    )

    return X.reshape(X.shape[0], -1), y


class COH_Preprocessing(Preprocessing):
    def __init__(self, path, f=5):
        super().__init__(path, f)

        self.data = np.hstack((self.data, np.zeros((len(self.data), 2))))

    def _get_epochs(self, time_per_array: float):
        epoch_size = int(time_per_array * 100)
        total_samples, num_channels = self.data.shape

        trimmed_data = self.data[: -(total_samples % epoch_size), 1:-2]
        self.X = trimmed_data.reshape(-1, epoch_size, num_channels - 3)

    def _get_psd(self):
        fs = 100
        window_size = fs
        overlap = window_size // 2

        num_epochs, _, num_channels = self.X.shape
        psd_array = np.empty(shape=(num_epochs, num_channels, 40))

        for epoch_idx, epoch_data in enumerate(self.X):
            for channel_idx, channel_data in enumerate(epoch_data.T):
                _, psd = signal.welch(
                    channel_data,
                    fs=fs,
                    window="hann",
                    nperseg=window_size,
                    noverlap=overlap,
                )
                psd_array[epoch_idx, channel_idx, :] = psd[1:41]

        self.X = psd_array

    def _get_coh(self):
        fs = 100
        window_size = fs
        overlap = window_size // 2

        num_epochs, _, num_channels = self.X.shape
        coherence_features = np.zeros(
            (num_epochs, (num_channels * (num_channels - 1) // 2) * 40)
        )

        # Function to process a single epoch
        def process_epoch(epoch_idx, epoch_data):
            feature_vector = []
            for i, channel_i in enumerate(epoch_data.T):
                for j, channel_j in enumerate(epoch_data.T[i + 1 :], start=i + 1):
                    _, coh_values = signal.coherence(
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

        # Use ThreadPoolExecutor to parallelize the processing
        with ThreadPoolExecutor() as executor:
            # Submit each epoch processing as a separate task
            futures = [
                executor.submit(process_epoch, epoch_idx, epoch_data)
                for epoch_idx, epoch_data in enumerate(self.X)
            ]

            # Wait for all futures to complete
            for future in futures:
                future.result()  # This blocks until the task is done

        self.X = coherence_features

    def get_subject_data(self, subject_index: int) -> tuple[np.ndarray, np.ndarray]:
        self._butter_bandpass_filter(fs=512, highcut=50)
        self._downsample()

        self.data = self.data[500:-500]

        self._get_epochs(10)

        # self._get_psd()
        self._get_coh()

        return self.X, np.ones(shape=len(self.X)) * subject_index


def get_coh_data(
    subjects: list[int], type: str, time: str
) -> tuple[np.ndarray, np.ndarray]:
    with ThreadPoolExecutor() as executor:
        data = list(
            executor.map(
                lambda i: COH_Preprocessing(
                    f"data/subject_0{i}_{type}_{time}.csv"
                ).get_subject_data(i - min(subjects)),
                subjects,
            )
        )

    X, y = np.vstack([data[i][0] for i in range(len(subjects))]), np.hstack(
        [data[i][1] for i in range(len(subjects))]
    )

    return X.reshape(X.shape[0], -1), y


if __name__ == "__main__":
    # X1, y1 = Preprocessing("data/subject_02_session_01.csv").get_subject_data(1)
    # X2, y2 = Preprocessing("data/subject_03_session_01.csv").get_subject_data(2)

    coh1 = COH_Preprocessing("data/subject_02_fixing_after.csv")
    coh2 = COH_Preprocessing("data/subject_03_fixing_after.csv")

    X1, y1 = coh1.get_subject_data(0)
    X2, y2 = coh2.get_subject_data(1)

    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)

    X = pca.fit(X)
    X1 = pca.transform(X1)
    X2 = pca.transform(X2)

    import matplotlib.pyplot as plt

    plt.scatter(X1[:, 0], X1[:, 1])
    plt.scatter(X2[:, 0], X2[:, 1])
    plt.show()
