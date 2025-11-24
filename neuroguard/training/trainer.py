"""
EEG Model Trainer
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Dict, Any
import logging
from datetime import datetime
import json
from pathlib import Path

from neuroguard.models.similarity.eegnet import EEGNetEmbeddingModel
from neuroguard.models.conv import EEGEmbedder
from neuroguard.commons.logger import setup_logger


class EEGTrainer:
    """
    EEG Model Trainer class
    """
    
    def __init__(
        self,
        model_name: str = "eegnet",
        batch_size: int = 32,
        learning_rate: float = 0.001,
        num_epochs: int = 10,
        checkpoint_freq: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        logger: Optional[logging.Logger] = None
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.checkpoint_freq = checkpoint_freq
        self.device = device
        self.logger = logger or setup_logger()
        
        # Initialize model
        self.model = self._init_model()
        self.model.to(self.device)
        
        # Initialize criterion and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def _init_model(self) -> nn.Module:
        """Initialize the model based on model_name."""
        if self.model_name.lower() == "eegnet":
            return EEGNetEmbeddingModel()
        elif self.model_name.lower() == "eegembedder":
            return EEGEmbedder()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def _load_data(self, data_path: str) -> tuple:
        """Load training data from file."""
        data_path = Path(data_path)

        # Handle directory containing FIF files
        if data_path.is_dir():
            X, y = self._load_fif_directory(data_path)
        else:
            # Load data based on extension
            if data_path.suffix == '.npz':
                data = np.load(data_path)
                X, y = data['X'], data['y']
            elif data_path.suffix in ['.pt', '.pth']:
                data = torch.load(data_path)
                X, y = data['X'], data['y']
            elif data_path.suffix == '.fif':
                X, y = self._load_fif_file(data_path)
            elif data_path.suffix == '.edf':
                X, y = self._load_edf_file(data_path)
            elif data_path.suffix in ['.csv', '.tsv']:
                X, y = self._load_csv_file(data_path)
            else:
                raise ValueError(f"Unsupported data format: {data_path.suffix}")

        # Convert to tensors if needed
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)

        # Reshape for EEGNet: (batch, channels, time) -> (batch, 1, channels, time)
        if len(X.shape) == 3:
            X = X.unsqueeze(1)  # Add the depth dimension for 2D convolutions

        # Create dataset
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        return dataloader, X.shape[0]

    def _load_fif_file(self, file_path: Path) -> tuple:
        """Load EEG data from a single FIF file."""
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
            # This is a simplified approach - you might need to segment based on other criteria
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

    def _load_fif_directory(self, dir_path: Path) -> tuple:
        """Load EEG data from a directory containing multiple FIF files."""
        import mne

        all_X = []
        all_y = []

        # Look for FIF files in the directory
        fif_files = list(dir_path.glob('*.fif')) + list(dir_path.glob('*.fif.gz'))

        if not fif_files:
            # If no FIF files, look for other formats in subdirectories
            # This might be for other data formats
            raise ValueError(f"No FIF files found in directory: {dir_path}")

        for fif_file in fif_files:
            try:
                # Extract labels from filename or use directory structure
                # This is a simplified approach - you might need to customize this
                # based on your specific file naming convention
                if 'participant' in fif_file.name or 'subject' in fif_file.name:
                    # Attempt to extract participant ID from filename
                    import re
                    match = re.search(r'(?:participant|subject|sub|suj_?)(\d+)', fif_file.name, re.IGNORECASE)
                    label = int(match.group(1)) if match else 0
                else:
                    # Use file index as label
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

    def _load_edf_file(self, file_path: Path) -> tuple:
        """Load EEG data from an EDF file."""
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

    def _load_csv_file(self, file_path: Path) -> tuple:
        """Load EEG data from a CSV file."""
        import pandas as pd

        df = pd.read_csv(file_path)

        # Assume the last column is the label and the rest are features
        # This might need adjustment based on your specific CSV format
        if 'label' in df.columns:
            y = df['label'].values
            X_cols = [col for col in df.columns if col != 'label']
            X = df[X_cols].values
        else:
            # If no explicit label column, use the last column
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values

        # Reshape X to 3D tensor if needed (batch, channels, time)
        # This is a simplified reshape - you might need to adapt based on your data structure
        if len(X.shape) == 2:
            # Assume X is (n_samples, n_features) and reshape to (n_samples, n_channels, n_time_points)
            # This is a basic reshape - you might need to adjust based on your actual data structure
            X = X.reshape(X.shape[0], 1, X.shape[1])  # Reshape to (batch, 1, features)

        return X, y.astype(int)
    
    def _train_epoch(self, train_loader: DataLoader) -> tuple:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)

            # Handle the case where model returns a tuple (embedding, classification_logits)
            if isinstance(output, tuple):
                embedding, classification_logits = output
                output = classification_logits
            else:
                classification_logits = output

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy
    
    def _validate(self, val_loader: DataLoader) -> tuple:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                # Handle the case where model returns a tuple (embedding, classification_logits)
                if isinstance(output, tuple):
                    embedding, classification_logits = output
                    output = classification_logits
                else:
                    classification_logits = output

                loss = self.criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy
    
    def train(self, data_path: str, save_path: str) -> None:
        """Train the model."""
        self.logger.info(f"Starting training for {self.model_name}")
        self.logger.info(f"Using device: {self.device}")
        
        # Load data
        train_loader, num_samples = self._load_data(data_path)
        self.logger.info(f"Loaded {num_samples} samples for training")
        
        # Split data for validation (use last 20% as validation)
        dataset_size = num_samples
        val_size = int(0.2 * dataset_size)
        train_size = dataset_size - val_size
        
        train_dataset = train_loader.dataset
        train_data, val_data = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)
        
        # Training loop
        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss, val_acc = self._validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            self.logger.info(
                f"Epoch {epoch}/{self.num_epochs}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            
            # Save checkpoint
            if epoch % self.checkpoint_freq == 0:
                self._save_checkpoint(epoch, save_path)
        
        # Save final model
        self._save_final_model(save_path)
        self._save_training_history(save_path)
        
        self.logger.info("Training completed successfully")
    
    def _save_checkpoint(self, epoch: int, save_path: str) -> None:
        """Save model checkpoint."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = save_path / f"checkpoint_epoch_{epoch}.pth"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
        }, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_final_model(self, save_path: str) -> None:
        """Save the final trained model."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        model_path = save_path / f"{self.model_name}_final.pth"
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'model_architecture': type(self.model).__name__,
        }, model_path)
        
        self.logger.info(f"Final model saved: {model_path}")
    
    def _save_training_history(self, save_path: str) -> None:
        """Save training history."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'model_name': self.model_name,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'training_date': datetime.now().isoformat()
        }
        
        history_path = save_path / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        self.logger.info(f"Training history saved: {history_path}")