import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from scipy import signal


class Preprocessing:
    def __init__(self, path: str, f: int = 5):
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


if __name__ == "__main__":
    X1, y1 = Preprocessing("data/subject_02_session_01.csv").get_subject_data(1)
    X2, y2 = Preprocessing("data/subject_03_session_01.csv").get_subject_data(2)
