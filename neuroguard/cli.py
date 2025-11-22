"""
EEG Library CLI - Command Line Interface for EEG model training and evaluation
"""
import argparse
import sys
import os
from typing import Any
import torch
import logging

# Add the project root to the path to import neuroguard modules
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from neuroguard.commons.logger import setup_logger
from neuroguard.training.trainer import EEGTrainer
from neuroguard.visualization.visualizer import EEGVisualizer
from neuroguard.evaluation.evaluator import EEGEvaluator


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser for the EEG library CLI."""
    parser = argparse.ArgumentParser(
        description="EEG Library - Command Line Interface for EEG model training and evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train an EEG model
  python -m neuroguard train --model eegnet --batch_size 32 --lr 0.001 --num_epochs 10
  
  # Visualize embeddings with t-SNE
  python -m neuroguard visualize --model_path ./models/eegnet.pth --method tsne --data_path ./data/
  
  # Evaluate a trained model
  python -m neuroguard evaluate --model_path ./models/eegnet.pth --test_data ./data/test/
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Training command
    train_parser = subparsers.add_parser("train", help="Train an EEG model")
    train_parser.add_argument("--model", type=str, default="eegnet", 
                              choices=["eegnet", "eegembedder"], help="Model architecture to use")
    train_parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train")
    train_parser.add_argument("--data_path", type=str, required=True, help="Path to training data (supports .npz, .pt/.pth, .fif, .edf, .csv, directories with FIF files)")
    train_parser.add_argument("--model_save_path", type=str, default="./models/", help="Path to save the trained model")
    train_parser.add_argument("--checkpoint_freq", type=int, default=5, help="Save checkpoint every N epochs")
    train_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                              help="Device to train on")
    train_parser.add_argument("--log_level", type=str, default="INFO", 
                              choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    
    # Visualization command
    visualize_parser = subparsers.add_parser("visualize", help="Visualize EEG embeddings")
    visualize_parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    visualize_parser.add_argument("--method", type=str, default="tsne", 
                                 choices=["tsne", "umap", "pca", "lda"], help="Visualization method")
    visualize_parser.add_argument("--data_path", type=str, required=True, help="Path to data for visualization (supports .npz, .pt/.pth, .fif, .edf, .csv, directories with FIF files)")
    visualize_parser.add_argument("--save_path", type=str, default="./plots/", help="Path to save visualization plots")
    visualize_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                                  help="Device to run inference on")
    visualize_parser.add_argument("--log_level", type=str, default="INFO", 
                                 choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    
    # Evaluation command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a trained EEG model")
    evaluate_parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    evaluate_parser.add_argument("--test_data", type=str, required=True, help="Path to test data (supports .npz, .pt/.pth, .fif, .edf, .csv, directories with FIF files)")
    evaluate_parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    evaluate_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                                 help="Device to run evaluation on")
    evaluate_parser.add_argument("--metrics", nargs="+", default=["accuracy"], 
                                 choices=["accuracy", "f1", "precision", "recall", "confusion_matrix"], 
                                 help="Metrics to compute during evaluation")
    evaluate_parser.add_argument("--save_results", type=str, default="./results/", help="Path to save evaluation results")
    evaluate_parser.add_argument("--log_level", type=str, default="INFO", 
                                 choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    
    return parser


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Set up logging
    logger = setup_logger(
        name="neuroguard_cli",
        log_level=getattr(logging, args.log_level.upper())
    )
    
    logger.info(f"Starting EEG Library CLI with command: {args.command}")
    
    try:
        if args.command == "train":
            trainer = EEGTrainer(
                model_name=args.model,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                num_epochs=args.num_epochs,
                checkpoint_freq=args.checkpoint_freq,
                device=args.device,
                logger=logger
            )
            trainer.train(data_path=args.data_path, save_path=args.model_save_path)
            
        elif args.command == "visualize":
            visualizer = EEGVisualizer(
                model_path=args.model_path,
                device=args.device,
                logger=logger
            )
            visualizer.generate_visualization(
                data_path=args.data_path,
                method=args.method,
                save_path=args.save_path
            )
            
        elif args.command == "evaluate":
            evaluator = EEGEvaluator(
                model_path=args.model_path,
                batch_size=args.batch_size,
                device=args.device,
                logger=logger
            )
            results = evaluator.evaluate(
                test_data_path=args.test_data,
                metrics=args.metrics
            )
            evaluator.save_results(results, args.save_results)
            
        else:
            logger.error(f"Unknown command: {args.command}")
            parser.print_help()
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()