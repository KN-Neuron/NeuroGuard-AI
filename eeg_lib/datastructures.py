from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Union
import torch
import numpy as np
import pandas as pd
from .types import EEGDataNDArray, EEGEpochsNDArray


@dataclass
class EEGData:
    """Represents EEG data with associated metadata."""
    epoch: EEGDataNDArray
    participant_id: str
    label: str
    timestamp: Optional[float] = None
    sampling_rate: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate that epoch has the expected shape."""
        if self.epoch.ndim != 2:
            raise ValueError(f"Epoch data must be 2D (channels, time_points), got shape {self.epoch.shape}")


@dataclass
class EEGParticipant:
    """Represents information about an EEG participant."""
    participant_id: str
    file: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EEGEpochs:
    """Container for multiple EEG epochs."""
    epochs: EEGEpochsNDArray  # Shape: (n_epochs, n_channels, n_time_points)
    participant_id: str
    labels: List[str]
    event_ids: Optional[Dict[str, int]] = None


@dataclass
class EEGPreprocessingConfig:
    """Configuration for EEG preprocessing."""
    lfreq: float = 1.0
    hfreq: float = 100.0
    notch_filter: Optional[List[int]] = None
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None
    tmin: float = 0.0
    tmax: float = 3.0
    microvolts_in_volt: int = 10**6

    def __post_init__(self) -> None:
        if self.notch_filter is None:
            self.notch_filter = [50]


@dataclass
class ModelConfig:
    """Configuration for EEG model parameters."""
    num_channels: int = 4
    num_classes: int = 4
    num_time_points: int = 751
    temporal_kernel_size: int = 32
    num_filters_first_layer: int = 16
    num_filters_second_layer: int = 32
    depth_multiplier: int = 2
    pool_kernel_size_1: int = 8
    pool_kernel_size_2: int = 16
    dropout_rate: float = 0.5
    max_norm_depthwise: float = 1.0
    max_norm_linear: float = 0.25
    embedding_dimension: int = 32


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    n_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    device: str = "cpu"
    num_workers: int = 0
    validation_split: float = 0.2
    random_state: int = 42
    log_every: int = 100


__all__ = [
    "EEGData",
    "EEGParticipant",
    "EEGEpochs",
    "EEGPreprocessingConfig",
    "ModelConfig",
    "TrainingConfig"
]