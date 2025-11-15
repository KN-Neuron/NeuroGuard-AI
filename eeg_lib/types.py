"""Type definitions for EEG data structures and tensor shapes."""
from typing import Any, TypeVar, Union
import numpy as np
import numpy.typing as npt
import torch


EEGTensor = TypeVar('EEGTensor', npt.NDArray[Any], torch.Tensor)
EEGDataNDArray = npt.NDArray[Any]  
EEGDataTensor = torch.Tensor  


EEGEmbeddingTensor = torch.Tensor  


EEGEpochsNDArray = npt.NDArray[Any]  
EEGEpochsTensor = torch.Tensor  


ModelInputTensor = torch.Tensor  
ModelOutputTensor = torch.Tensor  


EEGData = Union[npt.NDArray[Any], torch.Tensor]

__all__ = [
    "EEGTensor",
    "EEGDataNDArray",
    "EEGDataTensor",
    "EEGEmbeddingTensor",
    "EEGEpochsNDArray",
    "EEGEpochsTensor",
    "ModelInputTensor",
    "ModelOutputTensor",
    "EEGData"
]