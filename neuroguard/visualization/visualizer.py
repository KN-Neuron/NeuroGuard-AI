"""
EEG Visualization Module
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from umap import UMAP
from typing import Optional, Tuple
import logging
from pathlib import Path

from neuroguard.commons.logger import setup_logger


class EEGVisualizer:
    """
    EEG Visualization class for generating various plots from model embeddings
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        logger: Optional[logging.Logger] = None
    ):
        self.model_path = model_path
        self.device = device
        self.logger = logger or setup_logger()
        
        # Load the model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
    
    def _load_model(self):
        """Load the trained model."""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract model info
        if 'model_state_dict' in checkpoint:
            # For models saved with full state dict
            model_architecture = checkpoint.get('model_architecture', 'EEGNetEmbeddingModel')
            
            # Initialize the appropriate model based on saved architecture
            if model_architecture == 'EEGNetEmbeddingModel':
                from neuroguard.models.similarity.eegnet import EEGNetEmbeddingModel
                model = EEGNetEmbeddingModel()
            elif model_architecture == 'EEGEmbedder':
                from neuroguard.models.conv import EEGEmbedder
                model = EEGEmbedder()
            else:
                # Default to EEGNet if architecture unknown
                from neuroguard.models.similarity.eegnet import EEGNetEmbeddingModel
                model = EEGNetEmbeddingModel()
            
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # For models saved directly
            model = torch.load(self.model_path, map_location=self.device)
        
        return model
    
    def _extract_embeddings(self, data_loader) -> Tuple[np.ndarray, np.ndarray]:
        """Extract embeddings from the model."""
        embeddings = []
        labels = []
        
        with torch.no_grad():
            for batch_data, batch_labels in data_loader:
                batch_data = batch_data.to(self.device)
                batch_embeddings = self.model(batch_data)
                
                embeddings.append(batch_embeddings.cpu().numpy())
                labels.append(batch_labels.numpy())
        
        embeddings = np.vstack(embeddings)
        labels = np.hstack(labels)
        
        return embeddings, labels
    
    def _load_data(self, data_path: str):
        """Load data for visualization."""
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

        # Create dataset loader
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        return dataloader

    def _load_fif_file(self, file_path: Path):
        """Load EEG data from a single FIF file for visualization."""
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
        """Load EEG data from a directory containing multiple FIF files for visualization."""
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
        """Load EEG data from an EDF file for visualization."""
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
        """Load EEG data from a CSV file for visualization."""
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
    
    def _apply_dimensionality_reduction(
        self,
        embeddings: np.ndarray,
        method: str,
        n_components: int = 2
    ) -> np.ndarray:
        """Apply dimensionality reduction technique."""
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42)
        elif method.lower() == 'umap':
            reducer = UMAP(n_components=n_components, random_state=42)
        elif method.lower() == 'pca':
            reducer = PCA(n_components=n_components)
        elif method.lower() == 'lda':
            # For LDA, we need number of components to be min(n_features, n_classes - 1)
            n_classes = len(np.unique(self.labels))
            max_components = min(embeddings.shape[1], n_classes - 1, n_components)
            reducer = LinearDiscriminantAnalysis(n_components=max_components)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if method.lower() == 'lda':
            return reducer.fit_transform(embeddings, self.labels)
        else:
            return reducer.fit_transform(embeddings)
    
    def _plot_embeddings(
        self,
        reduced_embeddings: np.ndarray,
        labels: np.ndarray,
        method: str,
        save_path: str
    ) -> str:
        """Plot the reduced embeddings."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        
        # Create the plot
        scatter = plt.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            c=labels,
            cmap='tab10',
            alpha=0.7
        )
        
        plt.colorbar(scatter)
        plt.title(f'{method.upper()} Visualization of EEG Embeddings')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        
        # Save the plot
        plot_path = save_path / f"{method.lower()}_visualization.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"{method.upper()} visualization saved: {plot_path}")
        
        return str(plot_path)
    
    def generate_visualization(
        self,
        data_path: str,
        method: str = "tsne",
        save_path: str = "./plots/"
    ) -> str:
        """Generate visualization for the given data using the specified method."""
        self.logger.info(f"Starting {method.upper()} visualization")
        
        # Load data
        dataloader = self._load_data(data_path)
        self.logger.info("Data loaded successfully")
        
        # Extract embeddings
        self.logger.info("Extracting embeddings from model...")
        embeddings, labels = self._extract_embeddings(dataloader)
        self.labels = labels  # Store for potential LDA usage
        self.logger.info(f"Extracted embeddings of shape: {embeddings.shape}")
        
        # Apply dimensionality reduction
        self.logger.info(f"Applying {method.upper()} reduction...")
        reduced_embeddings = self._apply_dimensionality_reduction(embeddings, method)
        self.logger.info(f"Reduced embeddings shape: {reduced_embeddings.shape}")
        
        # Create plot
        plot_path = self._plot_embeddings(reduced_embeddings, labels, method, save_path)
        
        self.logger.info(f"Visualization completed: {plot_path}")
        
        return plot_path