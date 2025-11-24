import os
from typing import Optional, Tuple, List, Dict, Any
import numpy as np


import mne
import pandas as pd
from neuroguard.datastructures import EEGEpochs, EEGParticipant


class EEGDataExtractor:
    def __init__(
        self,
        data_dir: str,
        lfreq: float = 1.0,
        hfreq: float = 100.0,
        notch_filter: Optional[List[int]] = None,
        baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
        tmin: float = 0.0,
        tmax: float = 3.0,
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
        self.notch_filter = notch_filter if notch_filter is not None else [50]
        self.baseline = baseline
        self.tmin = tmin
        self.tmax = tmax

    def _read_from_dir(self) -> List[str]:
        """Returns a list of .fif files in the data directory."""
        return [f for f in os.listdir(self.data_dir) if f.endswith(".fif")]

    def _load_eeg(self) -> Tuple[List[EEGEpochs], List[EEGParticipant]]:
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

            microvolts_to_millivolts_conversion_factor = 10**-6
            eeg.apply_function(lambda x: x * microvolts_to_millivolts_conversion_factor)
            eeg.filter(l_freq=self.lfreq, h_freq=self.hfreq)
            eeg.notch_filter(self.notch_filter)
            if getattr(eeg, "annotations", None) is not None:
                print("Annotations summary:", eeg.annotations.description[:10])
            events, event_id = mne.events_from_annotations(eeg)
            if not event_id:
                print(f"No events found in file {file}")
                continue

            id_to_label = {v: k for k, v in event_id.items()}

            samples = events[:, 0]
            unique_vals, counts = np.unique(samples, return_counts=True)
            dup_samples = unique_vals[counts > 1]

            if dup_samples.size > 0:
                print(
                    f"Found {dup_samples.size} duplicated event sample indices in {file}."
                )
                priority = [
                    "RELAX_START",
                    "SENTENCE_START",
                    "RELAX_END",
                    "SENTENCE_END",
                ]
                kept_events = []
                for s in np.unique(samples):
                    rows = np.where(samples == s)[0]
                    if len(rows) == 1:
                        kept_events.append(events[rows[0]])
                        continue
                    codes = events[rows, 2].tolist()
                    labels = [id_to_label.get(c, str(c)) for c in codes]
                    chosen_idx = None
                    for p in priority:
                        if p in labels:
                            chosen_idx = rows[labels.index(p)]
                            break
                    if chosen_idx is None:
                        chosen_idx = rows[0]
                    present = list(zip(rows.tolist(), labels))
                    print(
                        f" sample {s} -> choices {present} -> keeping row {chosen_idx} (label {id_to_label[events[chosen_idx, 2]]})"
                    )
                    kept_events.append(events[chosen_idx])
                events = np.asarray(sorted(kept_events, key=lambda e: e[0]), dtype=int)
                print(
                    f" Reduced events to {len(events)} after resolving duplicates for file {file}."
                )

            present_codes = set(events[:, 2].tolist())
            filtered_event_id = {
                label: code for label, code in event_id.items() if code in present_codes
            }
            if len(filtered_event_id) != len(event_id):
                removed = set(event_id.keys()) - set(filtered_event_id.keys())
                print(
                    f"Removed from event_id (no events present after cleaning): {removed}"
                )

            epochs = mne.Epochs(
                raw=eeg,
                events=events,
                event_id=filtered_event_id,
                tmin=self.tmin,
                tmax=self.tmax,
                baseline=self.baseline,
                preload=True,
                event_repeated="drop",
            )
            numeric_labels = epochs.events[:, -1]
            labels = [id_to_label.get(l, "unknown") for l in numeric_labels]

            eeg_and_events.append(
                EEGEpochs(
                    epochs=epochs.get_data(),
                    participant_id=participant_id,
                    labels=labels,
                    event_ids=event_id,
                )
            )
            participants.append(
                EEGParticipant(participant_id=participant_id, file=file)
            )
        return eeg_and_events, participants

    def extract_dataframe(self) -> Tuple[pd.DataFrame, List[EEGParticipant]]:
        """
        Iterates over each participant's data, extracts each epoch as a numpy array,
        and returns a DataFrame with columns: participant_id, epoch, label.
        Also returns a list of participants with metadata.
        """
        eeg_and_events, participants = self._load_eeg()
        data = []
        for item in eeg_and_events:
            participant_id = item.participant_id
            epochs = item.epochs
            labels = item.labels
            epoch_data = epochs

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

    def extract_erp_dataframe(self) -> Tuple[pd.DataFrame, List[EEGParticipant]]:
        """
        Iterates over each participant's data, computes ERP by averaging epochs per label,
        and returns a DataFrame with columns: participant_id, label, erp.
        Also returns a list of participants with metadata.
        """
        eeg_and_events, participants = self._load_eeg()
        data = []

        for item in eeg_and_events:
            participant_id = item.participant_id
            epochs = item.epochs
            labels = item.labels
            epoch_data = epochs

            label_to_epochs: Dict[str, List[Any]] = {}
            for i, label in enumerate(labels):
                if label not in label_to_epochs:
                    label_to_epochs[label] = []
                label_to_epochs[label].append(epoch_data[i])

            for label, epoch_list in label_to_epochs.items():
                erp = np.mean(epoch_list, axis=0)
                erp = np.mean(erp, axis=0)
                data.append(
                    {"participant_id": participant_id, "label": label, "erp": erp}
                )

        df = pd.DataFrame(data)
        return df, participants


if __name__ == "__main__":
    DATA_DIR = "data/Kolory/"

    extractor = EEGDataExtractor(data_dir=DATA_DIR)
    eeg_df, participants_info = extractor.extract_dataframe()
