"""
Example script for training an EEG model
"""
import numpy as np
import os
from eeg_lib.training.trainer import EEGTrainer

# Create sample data
X = np.random.randn(200, 4, 751)  # 200 samples, 4 channels, 751 time points (EEGNet typical input)
y = np.random.randint(0, 4, size=200)  # 4 classes

# Save as .npz file
np.savez('sample_train_data.npz', X=X, y=y)

# Train the model
trainer = EEGTrainer(
    model_name='eegnet',
    batch_size=32,
    learning_rate=0.001,
    num_epochs=5,  # Using fewer epochs for the example
    checkpoint_freq=2
)

trainer.train(data_path='sample_train_data.npz', save_path='./models/')

print("Training completed! Model saved to ./models/")

# Clean up sample data
os.remove('sample_train_data.npz')