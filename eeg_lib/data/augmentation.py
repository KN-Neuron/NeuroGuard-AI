"""EEG data augmentation utilities."""
import torch
import numpy as np
from typing import Optional, Tuple, Union
from ..types import EEGDataTensor, EEGEpochsNDArray


def temporal_shift(
    data: Union[EEGDataTensor, EEGEpochsNDArray],
    shift_amount: float = 0.1
) -> Union[EEGDataTensor, EEGEpochsNDArray]:
    """
    Apply temporal shift augmentation by shifting the time series data.

    Args:
        data: EEG data tensor with shape (n_channels, n_time_points) or (batch_size, n_channels, n_time_points)
        shift_amount: Proportion of time points to shift (0.0 to 1.0)

    Returns:
        Augmented EEG data with same shape as input
    """
    shifted_data: Union[EEGDataTensor, EEGEpochsNDArray]
    if data.ndim == 2:
        n_channels, n_time_points = data.shape
        shift_samples = int(shift_amount * n_time_points)
        shift = np.random.randint(-shift_samples, shift_samples + 1)
        shifted_data = np.roll(data, shift, axis=1)
    elif data.ndim == 3:
        batch_size, n_channels, n_time_points = data.shape
        shift_samples = int(shift_amount * n_time_points)
        shifted_data = data.clone() if isinstance(data, torch.Tensor) else data.copy()
        for i in range(batch_size):
            shift = np.random.randint(-shift_samples, shift_samples + 1)
            shifted_data[i] = np.roll(shifted_data[i], shift, axis=1)  
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got shape {data.shape}")

    return shifted_data


def gaussian_noise(
    data: Union[EEGDataTensor, EEGEpochsNDArray], 
    noise_std: float = 0.01
) -> Union[EEGDataTensor, EEGEpochsNDArray]:
    """
    Add Gaussian noise to EEG data.
    
    Args:
        data: EEG data tensor with shape (n_channels, n_time_points) or (batch_size, n_channels, n_time_points)
        noise_std: Standard deviation of the Gaussian noise

    Returns:
        EEG data with added Gaussian noise
    """
    noise = np.random.normal(0, noise_std, data.shape)
    return data + noise


def scale_amplitude(
    data: Union[EEGDataTensor, EEGEpochsNDArray], 
    scale_factor: float = 0.1
) -> Union[EEGDataTensor, EEGEpochsNDArray]:
    """
    Scale EEG data amplitude by a random factor.
    
    Args:
        data: EEG data tensor with shape (n_channels, n_time_points) or (batch_size, n_channels, n_time_points)
        scale_factor: Maximum scaling factor (e.g., 0.1 means scale between 0.9 and 1.1)

    Returns:
        Scaled EEG data
    """
    scale = 1.0 + np.random.uniform(-scale_factor, scale_factor)
    return data * scale


class EEGAugmentation:
    """
    EEG data augmentation class that combines different augmentation techniques.
    """
    
    def __init__(
        self,
        temporal_shift_prob: float = 0.5,
        temporal_shift_amount: float = 0.1,
        gaussian_noise_prob: float = 0.5,
        gaussian_noise_std: float = 0.01,
        scaling_prob: float = 0.5,
        scaling_factor: float = 0.1
    ):
        """
        Initialize EEG augmentation parameters.
        
        Args:
            temporal_shift_prob: Probability of applying temporal shift
            temporal_shift_amount: Proportion of time points to shift (0.0 to 1.0)
            gaussian_noise_prob: Probability of adding Gaussian noise
            gaussian_noise_std: Standard deviation of Gaussian noise
            scaling_prob: Probability of scaling amplitude
            scaling_factor: Maximum scaling factor
        """
        self.temporal_shift_prob = temporal_shift_prob
        self.temporal_shift_amount = temporal_shift_amount
        self.gaussian_noise_prob = gaussian_noise_prob
        self.gaussian_noise_std = gaussian_noise_std
        self.scaling_prob = scaling_prob
        self.scaling_factor = scaling_factor

    def __call__(self, data: Union[EEGDataTensor, EEGEpochsNDArray]) -> Union[EEGDataTensor, EEGEpochsNDArray]:
        """
        Apply augmentation to EEG data.
        
        Args:
            data: EEG data tensor with shape (n_channels, n_time_points) or (batch_size, n_channels, n_time_points)
            
        Returns:
            Augmented EEG data with same shape as input
        """
        augmented_data = data.copy() if isinstance(data, np.ndarray) else data.clone()
        
        if np.random.random() < self.temporal_shift_prob:
            augmented_data = temporal_shift(augmented_data, self.temporal_shift_amount)
        
        if np.random.random() < self.gaussian_noise_prob:
            augmented_data = gaussian_noise(augmented_data, self.gaussian_noise_std)
        
        if np.random.random() < self.scaling_prob:
            augmented_data = scale_amplitude(augmented_data, self.scaling_factor)
        
        return augmented_data