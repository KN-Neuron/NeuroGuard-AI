import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np
import torch


@dataclass
class EEGConfig:
    """Configuration class for EEG-related constants and settings."""

    sampling_rate: int = 251
    num_of_classes: int = 2
    num_of_electrodes: int = 4
    standard_band_to_frequency_range: Optional[Dict[str, Tuple[int, int]]] = None
    numpy_data_type = np.float32
    x_tensor_data_type = torch.float32
    y_tensor_data_type = torch.long
    microvolts_in_volt: int = 10**6
    num_workers: int = os.cpu_count() or 1

    def __post_init__(self) -> None:
        if self.standard_band_to_frequency_range is None:
            self.standard_band_to_frequency_range = {
                "delta": (1, 4),
                "theta": (4, 8),
                "alpha": (8, 14),
                "beta": (14, 31),
                "gamma": (31, 49),
            }


eeg_config = EEGConfig()


SAMPLING_RATE = eeg_config.sampling_rate
NUM_OF_CLASSES = eeg_config.num_of_classes
NUM_OF_ELECTRODES = eeg_config.num_of_electrodes
STANDARD_BAND_TO_FREQUENCY_RANGE = eeg_config.standard_band_to_frequency_range
NUMPY_DATA_TYPE = eeg_config.numpy_data_type
X_TENSOR_DATA_TYPE = eeg_config.x_tensor_data_type
Y_TENSOR_DATA_TYPE = eeg_config.y_tensor_data_type
MICROVOLTS_IN_VOLT = eeg_config.microvolts_in_volt
NUM_WORKERS = eeg_config.num_workers


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

DATASETS_FOLDER: str = os.path.join(project_root, "neuroguard/data/datasets")
RESULTS_FOLDER: str = os.path.join(project_root, "neuroguard/data/result_summaries")
