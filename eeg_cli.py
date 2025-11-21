#!/usr/bin/env python3
"""
Main entry point for EEG Library CLI
"""
import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from eeg_lib.cli import main

if __name__ == "__main__":
    main()