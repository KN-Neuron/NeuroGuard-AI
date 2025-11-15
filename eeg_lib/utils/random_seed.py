"""Random seed utilities for reproducible results."""
import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducible results across different libraries.

    Args:
        seed: Random seed value (default 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # slower but reproducible