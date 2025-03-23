import numpy as np
from scipy.fftpack import dct
from scipy.signal import stft

bands = {
    'delta': (0, 1),    # 0.5-4 Hz (bin 1: 0-5 Hz)
    'theta': (1, 2),     # 4-8 Hz (bin 1-2: 5-10 Hz)
    'alpha': (2, 3),     # 8-13 Hz (bin 2-3: 10-15 Hz)
    'beta_low' : (3,4),
    'beta_mid': (4, 5),      # 13-30 Hz (bins 3-6: 15-30 Hz)
    'beta_high': (5,6),
    'gamma': (6, 10)     # 30-50 Hz (bins 6-10: 30-50 Hz)
}


def compute_band_energy(stft_magnitudes, bands):
    # Initialize output array: (n_channels, n_frames, n_bands)
    n_channels, n_frames, _ = stft_magnitudes.shape
    n_bands = len(bands)
    energy = np.zeros((n_channels, n_frames, n_bands))

    # Loop over channels, frames, and bands
    for c in range(n_channels):  # For each EEG channel (e.g., 4)
        for f_idx in range(n_frames):  # For each frame (e.g., 15)
            for b_idx, (start, end) in enumerate(bands.values()):  # For each band (e.g., 5)
                # Extract STFT magnitudes for this band
                band_magnitudes = stft_magnitudes[c, f_idx, start:end]
                # Compute energy: sum of squared magnitudes
                energy[c, f_idx, b_idx] = np.sum(band_magnitudes ** 2)

    return energy


def apply_dct(log_energy, n_coeffs=7):
    coeffs = np.zeros((log_energy.shape[0], log_energy.shape[1], n_coeffs))
    for c in range(log_energy.shape[0]):
        for f_idx in range(log_energy.shape[1]):
            coeffs[c, f_idx] = dct(log_energy[c, f_idx], norm='ortho')[:n_coeffs]
    return coeffs


# Full pipeline
def extract_features(eeg_sample, frame_length=50):
    # Truncate to 750 timesteps
    eeg_trunc = eeg_sample[:, :750]
    n_channels, n_timesteps = eeg_trunc.shape
    n_frames = n_timesteps // frame_length

    # Reshape into frames
    frames = eeg_trunc.reshape(n_channels, n_frames, frame_length)

    # STFT
    stft_magnitudes = np.zeros((n_channels, n_frames, frame_length // 2 + 1))
    for c in range(n_channels):
        for f_idx in range(n_frames):
            _, _, Zxx = stft(frames[c, f_idx], fs=250, nperseg=frame_length)

            stft_magnitudes[c, f_idx] = np.mean(np.abs(Zxx), axis=1)

    # Filter bank energy
    filter_energy = compute_band_energy(stft_magnitudes, bands)

    # Log + DCT
    log_energy = np.log(filter_energy + 1e-6)
    dct_coeffs = apply_dct(log_energy, n_coeffs=7)

    # Concatenate channels
    tdnn_input = dct_coeffs.transpose(1, 0, 2).reshape(n_frames, -1)
    return tdnn_input