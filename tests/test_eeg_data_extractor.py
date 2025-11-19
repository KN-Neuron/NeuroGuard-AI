import os
import unittest
from typing import cast
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

from eeg_lib.data.EEGDataExtractor import EEGDataExtractor


class TestEEGDataExtractor(unittest.TestCase):
    @patch("os.listdir")
    @patch("mne.io.read_raw_fif")
    @patch("mne.events_from_annotations")
    @patch("mne.Epochs")
    def test_extract_dataframe(
        self, mock_epochs, mock_events_from_annotations, mock_read_raw_fif, mock_listdir
    ):
        mock_listdir.return_value = ["dummy.fif"]

        dummy_raw = MagicMock()
        dummy_raw.pick_types.return_value = None
        dummy_raw.apply_function.return_value = None
        dummy_raw.filter.return_value = None
        dummy_raw.notch_filter.return_value = None
        mock_read_raw_fif.return_value = dummy_raw

        # symulujemy zdarzenia i mapowanie zdarzeń
        events = np.array([[0, 0, 1], [100, 0, 2]])
        event_id = {"red": 1, "green": 2}
        mock_events_from_annotations.return_value = (events, event_id)

        dummy_epochs = MagicMock()
        dummy_epochs.events = events
        # Symulujemy 2 epoki, np. każda epoka to sygnał 2-kanałowy z 3 punktami czasowymi.
        dummy_epochs.get_data.return_value = np.array(
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
        )
        mock_epochs.return_value = dummy_epochs

        extractor = EEGDataExtractor(data_dir="dummy_dir", tmin=0, tmax=1)
        df, participants = extractor.extract_dataframe()
        self.assertIsInstance(df, pd.DataFrame)
        df = cast(pd.DataFrame, df)

        self.assertIn("participant_id", df.columns)
        self.assertIn("epoch", df.columns)
        self.assertIn("label", df.columns)

        self.assertEqual(len(df), 2)

        self.assertEqual(len(participants), 1)
        self.assertEqual(participants[0]["participant_id"], "dummy")
        self.assertEqual(participants[0]["file"], "dummy.fif")


if __name__ == "__main__":
    unittest.main()
