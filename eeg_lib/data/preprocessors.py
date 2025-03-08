# preprocessing/signal_preprocessor.py
from commons.constant import SAMPLING_RATE


class EEGPreprocessor:
    def __init__(self, sampling_rate=SAMPLING_RATE):
        self.sampling_rate = sampling_rate
