import numpy as np
from pandas import read_csv
from scipy import signal

def downsample(X: np.ndarray, fs: float, f: int) -> np.ndarray:
    ratio = fs // f
    X_ = X[: X.shape[0] - X.shape[0] % ratio]
    return np.mean(X_.reshape(X_.shape[0] // ratio, ratio, -1), axis=1)


def butter_bandpass_filter(
    X: np.ndarray, fs: int = 512, lowcut: int = 1, highcut: int = 55, order: int = 4
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


def get_raw_rest_data(pre_path:str=".") -> tuple[list[np.ndarray], list[int]]:
    types = ["eyesclosed_before", "eyesclosed_after"]

    X = []
    y = []
    for i in range(2, 13):
        for t in types:
            data = read_csv(
                f"{pre_path}/data/subject_{str(i).zfill(2)}_{t}.csv", header=None
            ).to_numpy()[:, 1:]

            # x = butter_bandpass_filter(data, fs=512, highcut=50)
            x = downsample(data, 512, 100)

            sizes: np.ndarray = (
                np.random.dirichlet(np.ones(9) * 1000.0, size=1) * x.shape[0]
            )
            sizes = sizes.astype(np.int64).flatten()
            sizes[0] += x.shape[0] - sizes.sum()

            xs = np.array_split(x, np.cumsum(sizes))[:-1]
            xs = list(map(butter_bandpass_filter, xs))

            # X.extend(xs)
            X.extend(list(map(np.transpose, xs)))
            y.extend([i] * 9)
            
    return X, y


if __name__ == "__main__":
    # data = read_csv("data/subject_03_session_01.csv", header=None).to_numpy()
    # X1, y1 = extract_coh_features(data, 0)
    # data = read_csv("data/subject_03_session_02.csv", header=None).to_numpy()
    # X2, y2 = extract_coh_features(data, 1)
    # data = read_csv("data/subject_03_session_03.csv", header=None).to_numpy()
    # X3, y3 = extract_coh_features(data, 2)

    # X = np.vstack((X1, X2, X3))
    # y = np.hstack((y1, y2, y3))

    # from sklearn.linear_model import LogisticRegression
    # from sklearn.metrics import accuracy_score

    # print(X.shape)
    # print(y.shape)

    # reg = LogisticRegression()
    # reg.fit(X, y)

    # print(accuracy_score(y, reg.predict(X)))

    X, y = get_raw_rest_data()
    print(len(X))
    
