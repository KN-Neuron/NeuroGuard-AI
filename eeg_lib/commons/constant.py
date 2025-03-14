import os
import sys

import numpy as np
import torch

SAMPLING_RATE = 251
NUM_OF_CLASSES = 2
NUM_OF_ELECTRODES = 4
STANDARD_BAND_TO_FREQUENCY_RANGE = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 14),
    "beta": (14, 31),
    "gamma": (31, 49),
}

NUMPY_DATA_TYPE = np.float32
X_TENSOR_DATA_TYPE = torch.float32
Y_TENSOR_DATA_TYPE = torch.long

MICROVOLTS_IN_VOLT: int = 10**6

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

DATASETS_FOLDER: str = os.path.join(project_root, "eeg_lib/data/datasets")

NUM_WORKERS = os.cpu_count() or 1
