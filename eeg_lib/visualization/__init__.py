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

from eeg_lib.commons.logger import setup_logger


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
                from eeg_lib.models.similarity.eegnet import EEGNetEmbeddingModel
                model = EEGNetEmbeddingModel()
            elif model_architecture == 'EEGEmbedder':
                from eeg_lib.models.conv import EEGEmbedder
                model = EEGEmbedder()
            else:
                # Default to EEGNet if architecture is unknown
                from eeg_lib.models.similarity.eegnet import EEGNetEmbeddingModel
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
        # For now, support numpy format
        data_path = Path(data_path)
        
        if data_path.suffix == '.npz':
            data = np.load(data_path)
            X, y = data['X'], data['y']
        elif data_path.suffix == '.pt' or data_path.suffix == '.pth':
            data = torch.load(data_path)
            X, y = data['X'], data['y']
        else:
            raise ValueError(f"Unsupported data format: {data_path.suffix}")
        
        # Convert to tensors if needed
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)
        
        # Create dataset loader
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        return dataloader
    
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