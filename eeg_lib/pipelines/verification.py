if __name__ == "__main__":
    eegnet_model.eval()
    save_model(eegnet_model, "eegnet_triplet")

    # Update the data_dict with filtered data
    # fbc_data_dict = data_dict.copy()
    # fbc_data_dict['X_train'] = X_train_filtered
    # fbc_data_dict['X_val'] = X_val_filtered
    # fbc_data_dict['X_test'] = X_test_filtered
    #
    # fbcnet_loaders = create_data_loaders(fbc_data_dict)
    #
    # fbcnet_model = FBCNet(num_channels, num_samples, embedding_size, num_bands=9)
    #
    # fbcnet_optimizer = torch.optim.Adam(fbcnet_model.parameters(), lr=1e-3)
    #
    # fbcnet_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # fbcnet_model = fbcnet_model.to(device)

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
