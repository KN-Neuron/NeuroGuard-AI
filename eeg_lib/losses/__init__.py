"""Loss functions for EEG model training."""

from .triplet_loss import TripletLoss, OnlineTripletLoss

__all__ = ["TripletLoss", "OnlineTripletLoss"]
