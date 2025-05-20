import os
from typing import Any, Dict, List, Optional, Tuple

import mne
import pandas as pd


class EEGDataExtractor:
    def __init__(
        self,
        data_dir: str,
        lfreq: float = 1,
        hfreq: float = 100,
        notch_filter: list = [50],
        baseline: Optional[Tuple] = None,
        tmin: float = 0,
        tmax: float = 3,
    ):
        """
        Parameters:
            data_dir (str): Directory with .fif files.
            lfreq (float): Low cutoff frequency for bandpass filtering.
            hfreq (float): High cutoff frequency for bandpass filtering.
            notch_filter (list): Frequencies for notch filtering (e.g., to remove 50Hz line noise).
            baseline (tuple): Baseline correction period.
            tmin (float): Start time (in seconds) relative to the event.
            tmax (float): End time (in seconds) relative to the event.
        """
        self.data_dir = data_dir
        self.lfreq = lfreq
        self.hfreq = hfreq
        self.notch_filter = notch_filter
        self.baseline = baseline
        self.tmin = tmin
        self.tmax = tmax

    def _read_from_dir(self):
        """Returns a list of .fif files in the data directory."""
        return [f for f in os.listdir(self.data_dir) if f.endswith(".fif")]

    def _load_eeg(self):
        """
        Loads each .fif file, applies filtering, converts units,
        extracts events and epochs, and maps event codes to labels.
        """
        files = self._read_from_dir()
        eeg_and_events = []
        participants = []
        for file in files:
            participant_id = os.path.splitext(file)[0]
            file_path = os.path.join(self.data_dir, file)
            eeg = mne.io.read_raw_fif(file_path, preload=True)
            eeg.pick_types(eeg=True, stim=False, eog=False, exclude="bads")
            # Convert units (from µV to V if needed)
            eeg.apply_function(lambda x: x * 10**-6)
            eeg.filter(l_freq=self.lfreq, h_freq=self.hfreq)
            eeg.notch_filter(self.notch_filter)
            events, event_id = mne.events_from_annotations(eeg)
            if not event_id:
                print(f"No events found in file {file}")
                continue
            # reverse mapping: integer code -> label name (e.g., 1 -> 'red')
            id_to_label = {v: k for k, v in event_id.items()}
            epochs = mne.Epochs(
                raw=eeg,
                events=events,
                event_id=event_id,
                tmin=self.tmin,
                tmax=self.tmax,
                baseline=self.baseline,
                preload=True,
            )
            numeric_labels = epochs.events[:, -1]
            labels = [id_to_label.get(l, "unknown") for l in numeric_labels]
            eeg_and_events.append(
                {"participant_id": participant_id, "epochs": epochs, "labels": labels}
            )
            participants.append({"participant_id": participant_id, "file": file})
        return eeg_and_events, participants

    def extract_dataframe(self) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Iterates over each participant's data, extracts each epoch as a numpy array,
        and returns a DataFrame with columns: participant_id, epoch, label.
        Also returns a list of participants with metadata.
        """
        eeg_and_events, participants = self._load_eeg()
        data = []
        for item in eeg_and_events:
            participant_id = item["participant_id"]
            epochs = item["epochs"]
            labels = item["labels"]
            epoch_data = epochs.get_data()

            for i, label in enumerate(labels):
                single_epoch_data = epoch_data[i]
                data.append(
                    {
                        "participant_id": participant_id,
                        "epoch": single_epoch_data,
                        "label": label,
                    }
                )
        df = pd.DataFrame(data)
        return df, participants


if __name__ == "__main__":
    from eeg_lib.commons.constant import DATASETS_FOLDER

    DATA_DIR = f"{DATASETS_FOLDER}/Kolory/"

    extractor = EEGDataExtractor(data_dir=DATA_DIR)
    eeg_df, participants_info = extractor.extract_dataframe()
