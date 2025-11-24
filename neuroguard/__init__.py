from . import datastructures, types
from .commons import constant, logger
from .data import data_setup, encode_features_labels
from .losses import TripletLoss, OnlineTripletLoss
from .models.similarity.conv import EEGEmbedder
from .models.verification.eegnet import EEGNetEmbeddingModel

__version__ = "0.1.0"

__all__ = [
    "datastructures",
    "types",
    "constant",
    "logger",
    "data_setup",
    "encode_features_labels",
    "TripletLoss",
    "OnlineTripletLoss",
    "EEGEmbedder",
    "EEGNetEmbeddingModel",
]
