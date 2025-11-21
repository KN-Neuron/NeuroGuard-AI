# EEG Library

EEG Library is a comprehensive Python package for EEG embedding generation, model training, and evaluation. It provides a command-line interface for easy model training, visualization, and evaluation of EEG data.

## Features

- **Model Training**: Train various EEG models with configurable parameters
- **Embedding Generation**: Create high-quality EEG embeddings for various applications
- **Visualization**: Generate t-SNE, UMAP, PCA, and LDA visualizations of embeddings
- **Model Evaluation**: Comprehensive evaluation metrics including accuracy, F1, precision, recall, and confusion matrices
- **Checkpointing**: Automatic model checkpointing during training
- **CLI Interface**: Easy-to-use command-line interface for all operations

## Installation

```bash
pip install -e .
```

## Quick Start

First, prepare your EEG data in .npz format with 'X' as input and 'y' as labels:

```python
import numpy as np

# Example: Create sample data
X = np.random.randn(100, 4, 751)  # 100 samples, 4 channels, 751 time points
y = np.random.randint(0, 4, size=100)  # 4 classes
np.savez('sample_data.npz', X=X, y=y)
```

Then train a model:

```bash
python -m eeg_lib train \
  --model eegnet \
  --batch_size 32 \
  --lr 0.001 \
  --num_epochs 10 \
  --data_path ./sample_data.npz \
  --model_save_path ./models/ \
  --checkpoint_freq 5
```

## Usage

### Training a Model

```bash
python -m eeg_lib train \
  --model eegnet \
  --batch_size 32 \
  --lr 0.001 \
  --num_epochs 10 \
  --data_path ./data/train.npz \
  --model_save_path ./models/ \
  --checkpoint_freq 5
```

### Visualizing Embeddings

```bash
python -m eeg_lib visualize \
  --model_path ./models/eegnet_final.pth \
  --method tsne \
  --data_path ./data/test.npz \
  --save_path ./plots/
```

### Evaluating a Model

```bash
python -m eeg_lib evaluate \
  --model_path ./models/eegnet_final.pth \
  --test_data ./data/test.npz \
  --metrics accuracy f1 precision recall confusion_matrix \
  --save_results ./results/
```

## Command Line Interface

### Training Command

```bash
python -m eeg_lib train [OPTIONS]
```

- `--model`: Model architecture (eegnet, eegembedder) [default: eegnet]
- `--batch_size`: Batch size for training [default: 32]
- `--lr`: Learning rate [default: 0.001]
- `--num_epochs`: Number of epochs to train [default: 10]
- `--data_path`: Path to training data (required)
- `--model_save_path`: Path to save the trained model [default: ./models/]
- `--checkpoint_freq`: Save checkpoint every N epochs [default: 5]
- `--device`: Device to train on [default: cuda if available, else cpu]
- `--log_level`: Logging level [default: INFO]

### Visualization Command

```bash
python -m eeg_lib visualize [OPTIONS]
```

- `--model_path`: Path to trained model (required)
- `--method`: Visualization method (tsne, umap, pca, lda) [default: tsne]
- `--data_path`: Path to data for visualization (required)
- `--save_path`: Path to save visualization plots [default: ./plots/]
- `--device`: Device to run inference on [default: cuda if available, else cpu]
- `--log_level`: Logging level [default: INFO]

### Evaluation Command

```bash
python -m eeg_lib evaluate [OPTIONS]
```

- `--model_path`: Path to trained model (required)
- `--test_data`: Path to test data (required)
- `--batch_size`: Batch size for evaluation [default: 32]
- `--device`: Device to run evaluation on [default: cuda if available, else cpu]
- `--metrics`: Metrics to compute (accuracy, f1, precision, recall, confusion_matrix) [default: accuracy]
- `--save_results`: Path to save evaluation results [default: ./results/]
- `--log_level`: Logging level [default: INFO]

## Data Format

The library supports multiple data formats:

### Standard Formats
- **NumPy (.npz)**: With 'X' and 'y' keys for data and labels
- **PyTorch (.pt/.pth)**: With 'X' and 'y' keys for data and labels
- **CSV (.csv/.tsv)**: With optional 'label' column or last column as labels

### EEG-Specific Formats
- **FIF (.fif)**: MNE-compatible format with events for labels
- **EDF (.edf)**: European Data Format with events for labels
- **Directory**: FIF/EDF files in a directory, with labels extracted from filenames

#### Examples

**NumPy format:**
```python
import numpy as np
# Save as .npz
np.savez('data.npz', X=X_data, y=y_labels)
```

**PyTorch format:**
```python
import torch
# Save as .pt
torch.save({'X': X_data, 'y': y_labels}, 'data.pt')
```

**FIF format:**
```bash
# Use directory containing FIF files
python -m eeg_lib train --data_path ./eeg_data/fif_files/

# Or single FIF file
python -m eeg_lib train --data_path ./eeg_data/subject_01_raw.fif
```

**CSV format:**
```python
import pandas as pd
df = pd.DataFrame({
    'channel1': [...],  # EEG channel data
    'channel2': [...],
    'channel3': [...],
    'channel4': [...],
    'label': [...]      # Classification labels
})
df.to_csv('eeg_data.csv', index=False)
```

## Supported Models

- **EEGNet**: A convolutional neural network specifically designed for EEG data
- **EEGEmbedder**: A custom embedding model for generating EEG embeddings

## Examples

See the `examples/` directory for complete usage examples, including:
- `train_example.py` - Example script to train a model programmatically
- `QUICKSTART.md` - Detailed quick start guide

## License

MIT