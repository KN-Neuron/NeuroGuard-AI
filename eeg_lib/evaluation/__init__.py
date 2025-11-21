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
        
        if test_data_path.suffix == '.npz':
            data = np.load(test_data_path)
            X, y = data['X'], data['y']
        elif test_data_path.suffix == '.pt' or test_data_path.suffix == '.pth':
            data = torch.load(test_data_path)
            X, y = data['X'], data['y']
        else:
            raise ValueError(f"Unsupported data format: {test_data_path.suffix}")
        
        # Convert to tensors if needed
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)
        
        # Create dataset loader
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        return dataloader
    
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