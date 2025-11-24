"""
Simple test to verify the library structure
"""
try:
    from neuroguard import __version__
    print(f"EEG Library version: {__version__}")
    
    from neuroguard.training.trainer import EEGTrainer
    print("✓ Training module imported successfully")
    
    from neuroguard.visualization.visualizer import EEGVisualizer  
    print("✓ Visualization module imported successfully")
    
    from neuroguard.evaluation.evaluator import EEGEvaluator
    print("✓ Evaluation module imported successfully")
    
    from neuroguard.cli import create_parser
    parser = create_parser()
    print("✓ CLI parser created successfully")
    
    print("\nAll modules imported successfully! The library structure is correct.")
    
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Error: {e}")