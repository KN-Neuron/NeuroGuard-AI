import numpy as np
from scipy.fftpack import dct
from scipy.signal import stft
from scipy.signal import welch

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
    """
    Compute band-specific energy from STFT magnitudes.

    Parameters

    :param stft_magnitudes : np.ndarray - STFT magnitude array of shape (n_channels, n_frames, n_freq_bins).
    :param bands : dict - Mapping of band names to (start_bin, end_bin) indices for frequency bins.

    Returns
    energy : np.ndarray - Energy per channel, per frame, per band, shape (n_channels, n_frames, n_bands).
    """
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
    """
    Apply Discrete Cosine Transform (DCT) to log-energy features.

    Parameters
    :param log_energy : np.ndarray - Logarithm of filter bank energy, shape (n_channels, n_frames, n_bands).
    :param n_coeffs : int - Number of DCT coefficients to retain per band.

    Returns
    coeffs : np.ndarray - DCT coefficients, shape (n_channels, n_frames, n_coeffs).
    """
    coeffs = np.zeros((log_energy.shape[0], log_energy.shape[1], n_coeffs))
    for c in range(log_energy.shape[0]):
        for f_idx in range(log_energy.shape[1]):
            coeffs[c, f_idx] = dct(log_energy[c, f_idx], norm='ortho')[:n_coeffs]
    return coeffs


# Full pipeline
def extract_features(eeg_sample, frame_length=50, trunc=750, fs=250):
    """
    Extract TDNN-style features from a multichannel EEG sample.

    Steps:
      1. Truncate or pad to `trunc` timesteps.
      2. Split into non-overlapping frames of length `frame_length`.
      3. Compute STFT magnitudes for each frame.
      4. Aggregate energy in predefined bands.
      5. Apply log + DCT to obtain cepstral-like features.
      6. Flatten per-frame, per-channel features for TDNN input.

    Parameters
    :param eeg_sample : np.ndarray -Raw EEG data, shape (n_channels, n_times).
    :param frame_length : int - Number of samples per frame (e.g. 50 for 200ms at 250Hz).
    :param trunc : int -Number of timesteps to keep from the start (must be multiple of frame_length).
    :param fs : int - Sampling frequency in Hz.

    Returns
    tdnn_input : np.ndarray - Feature array of shape (n_frames, n_channels * n_dct_coeffs).
    """
    # Truncate to 750 timesteps
    eeg_trunc = eeg_sample[:, :trunc]
    n_channels, n_timesteps = eeg_trunc.shape
    n_frames = n_timesteps // frame_length

    # Reshape into frames
    frames = eeg_trunc.reshape(n_channels, n_frames, frame_length)

    # STFT
    stft_magnitudes = np.zeros((n_channels, n_frames, frame_length // 2 + 1))
    for c in range(n_channels):
        for f_idx in range(n_frames):
            _, _, Zxx = stft(frames[c, f_idx], fs=fs, nperseg=frame_length)

            stft_magnitudes[c, f_idx] = np.mean(np.abs(Zxx), axis=1)

    # Filter bank energy
    filter_energy = compute_band_energy(stft_magnitudes, bands)

    # Log + DCT
    log_energy = np.log(filter_energy + 1e-6)
    dct_coeffs = apply_dct(log_energy, n_coeffs=7)

    # Concatenate channels
    tdnn_input = dct_coeffs.transpose(1, 0, 2).reshape(n_frames, -1)
    return tdnn_input



def extract_psd_features(eeg_sample,
                         fs=250,
                         frame_length_s=1.0,
                         hop_length_s=0.5,
                         bands=bands):
    """
    Compute PSD-based band-power features in sliding windows.

    Parameters
    :param eeg_sample : np.ndarray -Raw EEG data, shape (n_channels, n_times).
    :param fs : int -Sampling rate in Hz.
    :param frame_length_s : float -Window duration in seconds for Welch PSD.
    :param hop_length_s : float - Hop (step) duration in seconds between windows.
    :param bands : dict -Frequency bands defined as (lo, hi) pairs.

    Returns
    out : np.ndarray - PSD band-power matrix of shape (n_frames, n_channels * n_bands).
    """
    n_channels, n_times = eeg_sample.shape
    nperseg = int(frame_length_s * fs)
    hop = int(hop_length_s * fs)

    # how many windows?
    n_frames = 1 + (n_times - nperseg) // hop
    n_bands = len(bands)
    out = np.zeros((n_frames, n_channels * n_bands), dtype=np.float32)

    # precompute fenband masks per channel (same for all channels)
    freqs, _ = welch(eeg_sample[0], fs=fs, nperseg=nperseg)
    band_masks = [(freqs >= lo) & (freqs < hi) for lo, hi in bands.values()]

    for f in range(n_frames):
        start = f * hop
        stop  = start + nperseg
        # for each channel, compute PSD and then band‑power
        frame_bp = []
        for ch in range(n_channels):
            _, psd = welch(eeg_sample[ch, start:stop],
                           fs=fs, nperseg=nperseg)
            # average each band
            for mask in band_masks:
                frame_bp.append(psd[mask].mean())
        out[f] = frame_bp

    return out  # shape (n_frames, n_channels * n_bands)