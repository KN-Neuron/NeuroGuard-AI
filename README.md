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
python -m neuroguard train \
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
python -m neuroguard train \
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
python -m neuroguard visualize \
  --model_path ./models/eegnet_final.pth \
  --method tsne \
  --data_path ./data/test.npz \
  --save_path ./plots/
```

### Evaluating a Model

```bash
python -m neuroguard evaluate \
  --model_path ./models/eegnet_final.pth \
  --test_data ./data/test.npz \
  --metrics accuracy f1 precision recall confusion_matrix \
  --save_results ./results/
```

## Command Line Interface

### Training Command

```bash
python -m neuroguard train [OPTIONS]
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
python -m neuroguard visualize [OPTIONS]
```

- `--model_path`: Path to trained model (required)
- `--method`: Visualization method (tsne, umap, pca, lda) [default: tsne]
- `--data_path`: Path to data for visualization (required)
- `--save_path`: Path to save visualization plots [default: ./plots/]
- `--device`: Device to run inference on [default: cuda if available, else cpu]
- `--log_level`: Logging level [default: INFO]

### Evaluation Command

```bash
python -m neuroguard evaluate [OPTIONS]
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
python -m neuroguard train --data_path ./eeg_data/fif_files/

# Or single FIF file
python -m neuroguard train --data_path ./eeg_data/subject_01_raw.fif
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

## Installation

### Prerequisites
- Python 3.12 or higher
- pip package manager

### Installing from Source

1. **Clone or download the repository:**
```bash
git clone https://github.com/your-username/NeuroGuard-AI.git
cd NeuroGuard-AI
```

2. **Install in development mode:**
```bash
pip install -e .
```

3. **Install with specific dependencies (optional):**
```bash
pip install -e .[dev]  # Includes development dependencies
```

### Installing in Your Own Project

You can install NeuroGuard as a dependency in your own project using several methods:

#### Method 1: Direct pip install from GitHub
```bash
pip install git+https://github.com/your-username/NeuroGuard-AI.git
```

#### Method 2: Add to your requirements.txt
```txt
git+https://github.com/your-username/NeuroGuard-AI.git
```

#### Method 3: Copy as a submodule
```bash
git submodule add https://github.com/your-username/NeuroGuard-AI.git
pip install -e ./NeuroGuard-AI
```

## Usage as a Library

### 1. Using the Command Line Interface (CLI)

The NeuroGuard library provides a complete command-line interface for EEG processing tasks:

#### Training Models
```bash
# Train with FIF files directory
python -m neuroguard train \
  --model eegnet \
  --batch_size 32 \
  --lr 0.001 \
  --num_epochs 10 \
  --data_path ./eeg_data/fif_files/ \
  --model_save_path ./models/ \
  --checkpoint_freq 5

# Train with numpy data
python -m neuroguard train \
  --model eegnet \
  --data_path ./data/train.npz \
  --model_save_path ./models/
```

#### Visualization
```bash
# Visualize embeddings from trained model
python -m neuroguard visualize \
  --model_path ./models/eegnet_final.pth \
  --data_path ./data/test.npz \
  --method tsne \
  --save_path ./plots/

# Visualize from FIF files
python -m neuroguard visualize \
  --model_path ./models/eegnet_final.pth \
  --data_path ./eeg_data/test_subjects/ \
  --method umap \
  --save_path ./plots/
```

#### Evaluation
```bash
# Evaluate model performance
python -m neuroguard evaluate \
  --model_path ./models/eegnet_final.pth \
  --test_data ./data/test.npz \
  --metrics accuracy f1 precision recall confusion_matrix \
  --save_results ./results/
```

### 2. Using Programmatically in Python

#### Training Programmatically
```python
from neuroguard.training.trainer import EEGTrainer

# Create and configure trainer
trainer = EEGTrainer(
    model_name='eegnet',
    batch_size=32,
    learning_rate=0.001,
    num_epochs=20,
    checkpoint_freq=5,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Train the model with FIF files
trainer.train(
    data_path='./eeg_data/fif_files/',
    save_path='./models/my_experiment/'
)
```

#### Visualization Programmatically
```python
from neuroguard.visualization.visualizer import EEGVisualizer

# Create visualizer
visualizer = EEGVisualizer(
    model_path='./models/eegnet_final.pth',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Generate visualization from FIF directory
plot_path = visualizer.generate_visualization(
    data_path='./eeg_data/fif_files/',
    method='tsne',
    save_path='./plots/my_visualization/'
)
```

#### Evaluation Programmatically
```python
from neuroguard.evaluation.evaluator import EEGEvaluator

# Create evaluator
evaluator = EEGEvaluator(
    model_path='./models/eegnet_final.pth',
    batch_size=32,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Evaluate with multiple metrics
results = evaluator.evaluate(
    test_data_path='./eeg_data/fif_files/',
    metrics=['accuracy', 'f1', 'precision', 'recall', 'confusion_matrix']
)

# Save results
evaluator.save_results(results, './results/my_evaluation/')
```

### 3. Data Format Support

NeuroGuard supports multiple EEG data formats:

#### FIF Files (Recommended for EEG data)
```python
# Directory with multiple FIF files
data_path = './eeg_data/subjects/'  # Multiple .fif files

# Single FIF file
data_path = './eeg_data/subject_01_raw.fif'
```

#### NumPy Format
```python
import numpy as np

# Save as .npz with 'X' and 'y' keys
data = {
    'X': np.random.randn(100, 4, 751),  # (samples, channels, time_points)
    'y': np.random.randint(0, 4, size=100)  # labels
}
np.savez('data.npz', **data)
```

#### CSV Format
```python
import pandas as pd

# CSV with channel columns and label column
df = pd.DataFrame({
    'ch1': [...],     # First EEG channel values
    'ch2': [...],     # Second EEG channel values
    'ch3': [...],     # Third EEG channel values
    'ch4': [...],     # Fourth EEG channel values
    'label': [...]    # Classification labels
})
df.to_csv('eeg_data.csv', index=False)
```

### 4. Model Architectures

#### EEGNet Model
- Optimized for motor imagery and P300 classification
- Efficient for limited EEG data
- Default choice for most EEG tasks

#### EEGEmbedder Model
- Custom embedding model for similarity tasks
- Good for verification and matching applications

## Examples

### Complete Pipeline Example
```python
import numpy as np
from neuroguard.training.trainer import EEGTrainer
from neuroguard.visualization.visualizer import EEGVisualizer
from neuroguard.evaluation.evaluator import EEGEvaluator

# Step 1: Train model with FIF data
trainer = EEGTrainer(model_name='eegnet', batch_size=32, num_epochs=10)
trainer.train(
    data_path='./eeg_data/fif_files/',
    save_path='./models/exp1/'
)

# Step 2: Visualize embeddings
visualizer = EEGVisualizer(model_path='./models/exp1/eegnet_final.pth')
visualizer.generate_visualization(
    data_path='./eeg_data/fif_files/',
    method='tsne',
    save_path='./plots/exp1/'
)

# Step 3: Evaluate model
evaluator = EEGEvaluator(model_path='./models/exp1/eegnet_final.pth')
results = evaluator.evaluate(
    test_data_path='./eeg_data/fif_files/',
    metrics=['accuracy', 'f1', 'confusion_matrix']
)
evaluator.save_results(results, './results/exp1/')
```

## License

MIT
