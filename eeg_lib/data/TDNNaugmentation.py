import numpy as np
import torch


def add_gaussian_noise(eeg, std=0.01):
    noise = np.random.normal(0, std, eeg.shape)
    return eeg + noise
