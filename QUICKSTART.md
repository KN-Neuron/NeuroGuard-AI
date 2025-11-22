# Quick Start Guide

This guide will help you get started with the EEG Library for training, visualizing, and evaluating EEG models.

## Installation

First, install the library:

```bash
pip install -e .
```

## Data Preparation

Prepare your EEG data in either `.npz` or `.pt` format with keys 'X' for input data and 'y' for labels:

```python
import numpy as np

# Example: Create sample data
X = np.random.randn(100, 4, 751)  # 100 samples, 4 channels, 751 time points
y = np.random.randint(0, 4, size=100)  # 4 classes

# Save to .npz file
np.savez('sample_data.npz', X=X, y=y)
```

## Training a Model

Train an EEG model with default parameters:

```bash
python -m neuroguard train \
  --model eegnet \
  --data_path ./sample_data.npz \
  --model_save_path ./models/
```

For more control over training parameters:

```bash
python -m neuroguard train \
  --model eegnet \
  --batch_size 16 \
  --lr 0.0001 \
  --num_epochs 20 \
  --data_path ./sample_data.npz \
  --model_save_path ./models/ \
  --checkpoint_freq 5
```

## Visualization

Visualize embeddings from a trained model:

```bash
python -m neuroguard visualize \
  --model_path ./models/eegnet_final.pth \
  --data_path ./sample_data.npz \
  --method tsne \
  --save_path ./plots/
```

Supported visualization methods:

- `tsne`: t-distributed Stochastic Neighbor Embedding
- `umap`: Uniform Manifold Approximation and Projection
- `pca`: Principal Component Analysis
- `lda`: Linear Discriminant Analysis

## Evaluation

Evaluate a trained model:

```bash
python -m neuroguard evaluate \
  --model_path ./models/eegnet_final.pth \
  --test_data ./sample_data.npz \
  --metrics accuracy f1 precision recall confusion_matrix \
  --save_results ./results/
```

## Expected Data Format

The library supports multiple formats for EEG data:

### Standard Formats

- **NumPy format (.npz):**

```python
import numpy as np
# Input data with shape (n_samples, n_channels, n_time_points)
# Labels with shape (n_samples,)
data = {
    'X': np.random.randn(100, 4, 751),
    'y': np.random.randint(0, 4, size=100)
}
np.savez('data.npz', **data)
```

- **PyTorch format (.pt/.pth):**

```python
import torch
# Input data with shape (n_samples, n_channels, n_time_points)
# Labels with shape (n_samples,)
data = {
    'X': torch.randn(100, 4, 751),
    'y': torch.randint(0, 4, size=(100,))
}
torch.save(data, 'data.pt')
```

### EEG-Specific Formats

- **FIF format:**

```bash
# Training with a directory of FIF files
python -m neuroguard train --data_path ./eeg_data/Kolory/

# Training with a single FIF file
python -m neuroguard train --data_path ./eeg_data/subject_01_raw.fif

# Visualization with FIF files
python -m neuroguard visualize --data_path ./eeg_data/test_subjects/ --method tsne
```

- **EDF format:**

```bash
# Same usage as FIF files
python -m neuroguard train --data_path ./eeg_data/eeg_data.edf
```

- **CSV format:**

```python
import pandas as pd
# Create EEG dataset with multiple channels
df = pd.DataFrame({
    'ch1': [...],     # First EEG channel values
    'ch2': [...],     # Second EEG channel values
    'ch3': [...],     # Third EEG channel values
    'ch4': [...],     # Fourth EEG channel values
    'label': [...]    # Classification labels
})
df.to_csv('eeg_data.csv', index=False)
```

## Checkpointing

The library automatically saves model checkpoints during training based on the `--checkpoint_freq` parameter. This allows you to resume training or use intermediate models.

## Model Saving

Models are saved in PyTorch format with the following structure:

- Final model: `{model_name}_final.pth`
- Checkpoints: `checkpoint_epoch_{N}.pth`
- Training history: `training_history.json`

## Next Steps

- Explore the `examples/` directory for more detailed examples
- Experiment with different model architectures
- Adjust hyperparameters to optimize for your specific dataset
- Use the visualization tools to understand your embeddings
