"""
Example usage of NeuroGuard-AI EEG processing library.

This example demonstrates basic usage of the EEG processing library,
including loading data, using models, and training.
"""

import torch
import numpy as np
from eeg_lib.models import EEGNetEmbeddingModel
from eeg_lib.losses import TripletLoss
from eeg_lib.datastructures import EEGData, ModelConfig, TrainingConfig
from eeg_lib.data import EEGNetColorDataset
from eeg_lib.commons import eeg_config


def create_sample_data():
    """Create sample EEG data for demonstration."""

    num_samples = 20
    epochs = []
    labels = []

    for i in range(num_samples):

        epoch = np.random.randn(4, 751).astype(np.float32)
        epochs.append(epoch)

        label = f"color_{i % 4}"
        labels.append(label)

    import pandas as pd

    df = pd.DataFrame({"epoch": epochs, "label": labels})

    return df


def main():
    """Demonstrate basic EEG processing workflow."""
    print("NeuroGuard-AI: Basic Usage Example")
    print("=" * 40)

    df = create_sample_data()
    print(f"Created sample data with {len(df)} samples")

    dataset = EEGNetColorDataset(df)
    print(f"Dataset created with {len(dataset)} samples")

    model_config = ModelConfig(
        num_channels=4, num_classes=4, num_time_points=751, embedding_dimension=32
    )

    model = EEGNetEmbeddingModel(
        num_channels=model_config.num_channels,
        num_classes=model_config.num_classes,
        num_time_points=model_config.num_time_points,
        embedding_dimension=model_config.embedding_dimension,
    )

    print(
        f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    criterion = TripletLoss(margin=1.0)
    print(f"Loss function initialized: {type(criterion).__name__}")

    dummy_input = torch.randn(1, 1, 4, 751)
    embeddings, logits = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Logits shape: {logits.shape}")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
