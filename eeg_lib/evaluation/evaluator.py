"""
EEG Model Evaluator
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Dict, List, Any
import logging
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from eeg_lib.commons.logger import setup_logger


class EEGEvaluator:
    """
    EEG Model Evaluator class
    """
    
    def __init__(
        self,
        model_path: str,
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        logger: Optional[logging.Logger] = None
    ):
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = device
        self.logger = logger or setup_logger()
        
        # Load the model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize criterion
        self.criterion = nn.CrossEntropyLoss()
    
    def _load_model(self):
        """Load the trained model."""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract model info
        if 'model_state_dict' in checkpoint:
            # For models saved with full state dict
            model_architecture = checkpoint.get('model_architecture', 'EEGNetEmbeddingModel')
            
            # Initialize the appropriate model based on saved architecture
            if model_architecture == 'EEGNetEmbeddingModel':
                from eeg_lib.models.similarity.eegnet import EEGNetEmbeddingModel
                model = EEGNetEmbeddingModel()
            elif model_architecture == 'EEGEmbedder':
                from eeg_lib.models.conv import EEGEmbedder
                model = EEGEmbedder()
            else:
                # Default to EEGNet if architecture unknown
                from eeg_lib.models.similarity.eegnet import EEGNetEmbeddingModel
                model = EEGNetEmbeddingModel()
            
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # For models saved directly
            model = torch.load(self.model_path, map_location=self.device)
        
        return model
    
    def _load_test_data(self, test_data_path: str):
        """Load test data."""
        test_data_path = Path(test_data_path)

        # Handle directory containing FIF files
        if test_data_path.is_dir():
            X, y = self._load_fif_directory(test_data_path)
        else:
            # Load data based on extension
            if test_data_path.suffix == '.npz':
                data = np.load(test_data_path)
                X, y = data['X'], data['y']
            elif test_data_path.suffix in ['.pt', '.pth']:
                data = torch.load(test_data_path)
                X, y = data['X'], data['y']
            elif test_data_path.suffix == '.fif':
                X, y = self._load_fif_file(test_data_path)
            elif test_data_path.suffix == '.edf':
                X, y = self._load_edf_file(test_data_path)
            elif test_data_path.suffix in ['.csv', '.tsv']:
                X, y = self._load_csv_file(test_data_path)
            else:
                raise ValueError(f"Unsupported data format: {test_data_path.suffix}")

        # Convert to tensors if needed
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)

        # Reshape for EEGNet: (batch, channels, time) -> (batch, 1, channels, time)
        if len(X.shape) == 3:
            X = X.unsqueeze(1)  # Add the depth dimension for 2D convolutions

        # Create dataset loader
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        return dataloader

    def _load_fif_file(self, file_path: Path):
        """Load EEG data from a single FIF file for evaluation."""
        import mne

        # Load the raw data
        raw = mne.io.read_raw_fif(file_path, preload=True)

        # Get events (assuming there are events in the file for labels)
        events = mne.find_events(raw)

        # Create epochs if there are events
        if len(events) > 0:
            # Define event IDs for your specific experiment
            event_ids = {str(e[2]): e[2] for e in events}
            epochs = mne.Epochs(raw, events, event_ids, tmin=0, tmax=3.0, baseline=None, preload=True, picks='eeg')

            # Extract data and labels
            X = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
            y = epochs.events[:, -1]  # Event IDs as labels

            # Adjust the number of channels for EEGNet (typically use first 4 channels or average/interpolate)
            n_epochs, n_channels, n_times = X.shape

            # If we have more than 4 channels, we can select the first 4
            if n_channels >= 4:
                X = X[:, :4, :]  # Take first 4 channels
            # If we have fewer than 4 channels, pad with zeros or repeat
            elif n_channels < 4:
                X_padded = np.zeros((n_epochs, 4, n_times))
                X_padded[:, :n_channels, :] = X
                X = X_padded

        else:
            # If no events, just return the raw data as continuous segments
            sfreq = raw.info['sfreq']
            duration = 3.0  # Use 3 seconds to match typical EEGNet input (750 time points at 250 Hz)
            n_samples = int(sfreq * duration)

            # Select only EEG channels
            picks = mne.pick_types(raw.info, eeg=True)
            raw_data = raw.get_data(picks=picks)
            n_channels, total_samples = raw_data.shape

            # Segment into epochs
            n_epochs = total_samples // n_samples
            X = np.zeros((n_epochs, min(4, n_channels), n_samples))
            y = np.zeros(n_epochs, dtype=int)  # Placeholder labels

            for i in range(n_epochs):
                start_idx = i * n_samples
                end_idx = start_idx + n_samples
                if n_channels >= 4:
                    # Take first 4 channels
                    X[i] = raw_data[:4, start_idx:end_idx]
                else:
                    # Use available channels, pad if needed
                    X[i, :n_channels, :] = raw_data[:, start_idx:end_idx]

        return X, y

    def _load_fif_directory(self, dir_path: Path):
        """Load EEG data from a directory containing multiple FIF files for evaluation."""
        import mne

        all_X = []
        all_y = []

        # Look for FIF files in the directory
        fif_files = list(dir_path.glob('*.fif')) + list(dir_path.glob('*.fif.gz'))

        if not fif_files:
            # If no FIF files, look for other formats
            raise ValueError(f"No FIF files found in directory: {dir_path}")

        for fif_file in fif_files:
            try:
                # Extract labels from filename or use directory structure
                if 'participant' in fif_file.name or 'subject' in fif_file.name:
                    import re
                    match = re.search(r'(?:participant|subject|sub|suj_?)(\d+)', fif_file.name, re.IGNORECASE)
                    label = int(match.group(1)) if match else 0
                else:
                    label = len(all_X)

                # Load the specific FIF file
                X_file, _ = self._load_fif_file(fif_file)

                # Assign the same label to all samples in this file
                y_file = np.full(X_file.shape[0], label, dtype=int)

                all_X.append(X_file)
                all_y.append(y_file)

            except Exception as e:
                self.logger.warning(f"Could not load {fif_file}: {str(e)}")
                continue

        if not all_X:
            raise ValueError(f"No valid FIF files could be loaded from {dir_path}")

        # Concatenate all data
        X = np.vstack(all_X)
        y = np.hstack(all_y)

        return X, y

    def _load_edf_file(self, file_path: Path):
        """Load EEG data from an EDF file for evaluation."""
        import mne

        # Load the EDF file
        raw = mne.io.read_raw_edf(file_path, preload=True)

        # Similar processing as FIF files
        events = mne.find_events(raw)

        if len(events) > 0:
            event_ids = {str(e[2]): e[2] for e in events}
            epochs = mne.Epochs(raw, events, event_ids, tmin=0, tmax=1.0, baseline=None, preload=True)
            X = epochs.get_data()
            y = epochs.events[:, -1]
        else:
            # Fallback to continuous data segmentation
            sfreq = raw.info['sfreq']
            duration = 1.0
            n_samples = int(sfreq * duration)

            raw_data = raw.get_data()
            n_channels, total_samples = raw_data.shape

            n_epochs = total_samples // n_samples
            X = np.zeros((n_epochs, n_channels, n_samples))
            y = np.zeros(n_epochs, dtype=int)

            for i in range(n_epochs):
                start_idx = i * n_samples
                end_idx = start_idx + n_samples
                X[i] = raw_data[:, start_idx:end_idx]

        return X, y

    def _load_csv_file(self, file_path: Path):
        """Load EEG data from a CSV file for evaluation."""
        import pandas as pd

        df = pd.read_csv(file_path)

        # Assume the last column is the label and the rest are features
        if 'label' in df.columns:
            y = df['label'].values
            X_cols = [col for col in df.columns if col != 'label']
            X = df[X_cols].values
        else:
            # If no explicit label column, use the last column
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values

        # Reshape X to 3D tensor if needed (batch, channels, time)
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])  # Reshape to (batch, 1, features)

        return X, y.astype(int)
    
    def evaluate(
        self,
        test_data_path: str,
        metrics: List[str] = ["accuracy"]
    ) -> Dict[str, Any]:
        """Evaluate the model on test data."""
        self.logger.info("Starting model evaluation")
        
        # Load test data
        test_loader = self._load_test_data(test_data_path)
        self.logger.info("Test data loaded successfully")
        
        # Evaluate the model
        predictions = []
        true_labels = []
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                pred = output.argmax(dim=1, keepdim=True)
                predictions.extend(pred.cpu().numpy().flatten())
                true_labels.extend(target.cpu().numpy())
        
        # Calculate metrics
        results = {}
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        avg_loss = total_loss / len(test_loader)
        results['loss'] = avg_loss
        
        if 'accuracy' in metrics:
            results['accuracy'] = accuracy_score(true_labels, predictions)
        
        if 'f1' in metrics:
            results['f1'] = f1_score(true_labels, predictions, average='weighted')
        
        if 'precision' in metrics:
            results['precision'] = precision_score(true_labels, predictions, average='weighted')
        
        if 'recall' in metrics:
            results['recall'] = recall_score(true_labels, predictions, average='weighted')
        
        if 'confusion_matrix' in metrics:
            cm = confusion_matrix(true_labels, predictions)
            results['confusion_matrix'] = cm.tolist()  # Convert to list for JSON serialization
        
        self.logger.info("Evaluation completed")
        for metric, value in results.items():
            if metric != 'confusion_matrix':
                self.logger.info(f"{metric}: {value}")
        
        return results
    
    def save_results(self, results: Dict[str, Any], save_path: str) -> str:
        """Save evaluation results to file."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        results_path = save_path / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)  # default=str for numpy types
        
        # Create and save confusion matrix plot if available
        if 'confusion_matrix' in results:
            cm_path = save_path / "confusion_matrix.png"
            cm = np.array(results['confusion_matrix'])
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"Evaluation results saved: {results_path}")
        
        return str(results_path)