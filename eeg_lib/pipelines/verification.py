from eeg_lib.commons.constant import DATASETS_FOLDER
from eeg_lib.data.data_loader.EEGDataExtractor import EEGDataExtractor
from eeg_lib.commons.constant import NUM_OF_CLASSES, NUM_OF_ELECTRODES
from eeg_lib.models.similarity.eegnet import EEGNet
from eeg_lib.models.similarity.fbcnet import FBCNet
from eeg_lib.utils.helpers import prepare_eeg_data, create_data_loaders, apply_bandpass_filters, save_model
from eeg_lib.utils.visualisations import plot_embedding_tsne, plot_loss_and_accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset, DataLoader


def train_epoch(model, data_loader, optimizer, loss_fn, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.train()
    model.to(device)
    running_loss = 0.0

    for batch in data_loader:
        anchor = batch['anchor'].to(device)
        positive = batch['positive'].to(device)
        negative = batch['negative'].to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)

        # Calculate loss
        loss = loss_fn(anchor_emb, positive_emb, negative_emb)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(data_loader)


def validate_epoch(model, data_loader, loss_fn, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.eval()
    model.to(device)
    running_loss = 0.0

    with torch.no_grad():  # No gradient computation during validation
        for batch in data_loader:
            anchor = batch['anchor'].to(device)
            positive = batch['positive'].to(device)
            negative = batch['negative'].to(device)

            # Forward pass
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)

            # Calculate loss
            loss = loss_fn(anchor_emb, positive_emb, negative_emb)

            running_loss += loss.item()

    return running_loss / len(data_loader)

if __name__ == "__main__":
    # Load data
    DATA_DIR = f"{DATASETS_FOLDER}/Kolory/"

    extractor = EEGDataExtractor(data_dir=DATA_DIR)
    eeg_df, participants_info = extractor.extract_dataframe()

    # Set manual seed for reproducibility.
    torch.manual_seed(42)

    # Prepare the data
    data_dict = prepare_eeg_data(eeg_df)

    # For FBCNet, apply bandpass filters
    X_train_filtered = apply_bandpass_filters(data_dict['X_train'])
    X_val_filtered = apply_bandpass_filters(data_dict['X_val'])
    X_test_filtered = apply_bandpass_filters(data_dict['X_test'])

    # Update the data_dict with filtered data
    fbc_data_dict = data_dict.copy()
    fbc_data_dict['X_train'] = X_train_filtered
    fbc_data_dict['X_val'] = X_val_filtered
    fbc_data_dict['X_test'] = X_test_filtered

    # Create data loaders for EEGNet
    eegnet_loaders = create_data_loaders(data_dict)

    # Create data loaders for FBCNet
    fbcnet_loaders = create_data_loaders(fbc_data_dict)

    batch_size = 16
    num_channels = NUM_OF_ELECTRODES
    num_samples = eeg_df['epoch'].iloc[0].shape[1]
    embedding_size = 32

    # Instantiate models
    eegnet_model = EEGNet(num_channels, num_samples, embedding_size)
    # fbcnet_model = FBCNet(num_channels, num_samples, embedding_size, num_bands=9)

    # Define triplet margin loss and optimizers
    triplet_loss = torch.nn.TripletMarginLoss(margin=0.05)
    eegnet_optimizer = torch.optim.Adam(eegnet_model.parameters(), lr=1e-3)
    # fbcnet_optimizer = torch.optim.Adam(fbcnet_model.parameters(), lr=1e-3)


    num_epochs = 10
    eegnet_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    fbcnet_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move models to device
    eegnet_model = eegnet_model.to(device)
    # fbcnet_model = fbcnet_model.to(device)

    # Training EEGNet
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(
            eegnet_model,
            eegnet_loaders['triplet']['train'],
            eegnet_optimizer,
            triplet_loss
        )

        # Validation
        val_loss = validate_epoch(
            eegnet_model,
            eegnet_loaders['triplet']['val'],
            triplet_loss
        )

        # Calculate a simple accuracy metric: percentage of triplets where
        # distance(anchor, positive) < distance(anchor, negative)
        correct = 0
        total = 0
        eegnet_model.eval()
        with torch.no_grad():
            for batch in eegnet_loaders['triplet']['val']:
                anchor = batch['anchor'].to(device)
                positive = batch['positive'].to(device)
                negative = batch['negative'].to(device)

                # Forward pass
                anchor_emb = eegnet_model(anchor)
                positive_emb = eegnet_model(positive)
                negative_emb = eegnet_model(negative)

                # Calculate distances
                pos_dist = F.pairwise_distance(anchor_emb, positive_emb)
                neg_dist = F.pairwise_distance(anchor_emb, negative_emb)

                # Count correct predictions (where positive sample is closer than negative)
                correct += torch.sum(pos_dist < neg_dist).item()
                total += anchor_emb.size(0)

        val_acc = correct / total if total > 0 else 0

        # Update history
        eegnet_history['train_loss'].append(train_loss)
        eegnet_history['val_loss'].append(val_loss)
        eegnet_history['train_acc'].append(0.0)  # We don't calculate training accuracy here
        eegnet_history['val_acc'].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, EEGNet Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Training FBCNet
    # for epoch in range(num_epochs):
    #     # Train
    #     train_loss = train_epoch(
    #         fbcnet_model,
    #         fbcnet_loaders['triplet']['train'],
    #         fbcnet_optimizer,
    #         triplet_loss
    #     )
    #
    #     # Validation
    #     val_loss = validate_epoch(
    #         fbcnet_model,
    #         fbcnet_loaders['triplet']['val'],
    #         triplet_loss
    #     )
    #
    #     # Calculate a simple accuracy metric: percentage of triplets where
    #     # distance(anchor, positive) < distance(anchor, negative)
    #     correct = 0
    #     total = 0
    #     eegnet_model.eval()
    #     with torch.no_grad():
    #         for batch in eegnet_loaders['triplet']['val']:
    #             anchor = batch['anchor'].to(device)
    #             positive = batch['positive'].to(device)
    #             negative = batch['negative'].to(device)
    #
    #             # Forward pass
    #             anchor_emb = fbcnet_model(anchor)
    #             positive_emb = fbcnet_model(positive)
    #             negative_emb = fbcnet_model(negative)
    #
    #             # Calculate distances
    #             pos_dist = F.pairwise_distance(anchor_emb, positive_emb)
    #             neg_dist = F.pairwise_distance(anchor_emb, negative_emb)
    #
    #             # Count correct predictions (where positive sample is closer than negative)
    #             correct += torch.sum(pos_dist < neg_dist).item()
    #             total += anchor_emb.size(0)
    #
    #     val_acc = correct / total if total > 0 else 0
    #
    #     # Update history
    #     fbcnet_history['train_loss'].append(train_loss)
    #     fbcnet_history['val_loss'].append(val_loss)
    #     fbcnet_history['train_acc'].append(0.0)  # We don't calculate training accuracy here
    #     fbcnet_history['val_acc'].append(val_acc)
    #
    #     print(f"Epoch {epoch + 1}/{num_epochs}, EEGNet Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Extract embeddings for visualization
    eegnet_model.eval()
    save_model(eegnet_model, "eegnet_triplet")

