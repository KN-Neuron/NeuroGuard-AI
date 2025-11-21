from ..data.data_loader.EEGDataExtractor import EEGDataExtractor
from .conv_training import train_triplet_online
from .embedding_generation import extract_embeddings_online
from ..utils.random_seed import set_seed
from ..data.encode_features_labels import encode_features_and_labels
from ..models.conv import EEGEmbedder
import torch
from torch import nn
import torch.optim as optim
from ..data.subject_based_split import split_by_user
from ..data.subject_based_split import create_online_dataloaders
import json
import os
import numpy as np


def save_result_to_json(result, json_path="grid_search_results.json"):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(result)

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def grid_search(
    seed: int,
    emb_dims: list[int],
    lfreqs: list[float],
    hfreqs: list[float],
    notch_filters: list[float],
    tmins: list[float],
    tmaxs: list[float],
    n_epochs_list: list[int],
    margins: list[float],
    destination_file: str = "grid_search_results.json'",
):
    combinations = (
        len(emb_dims)
        * len(lfreqs)
        * len(hfreqs)
        * len(notch_filters)
        * len(tmins)
        * len(tmaxs)
        * len(n_epochs_list)
        * len(margins)
    )
    current_iteration = 0
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for lfreq in lfreqs:
        for hfreq in hfreqs:
            for notch_filter in notch_filters:
                for tmin in tmins:
                    for tmax in tmaxs:

                        extractor = EEGDataExtractor(
                            data_dir="data/Kolory",
                            lfreq=lfreq,
                            hfreq=hfreq,
                            notch_filter=notch_filter,
                            tmin=tmin,
                            tmax=tmax,
                        )

                        df, participants = extractor.extract_dataframe()
                        df_labels = df["participant_id"]
                        X, y, le = encode_features_and_labels(df)
                        X_train, y_train, X_test, y_test, train_labels, test_labels = (
                            split_by_user(X, y, df_labels, random_state=seed)
                        )
                        train_loader, test_loader = create_online_dataloaders(
                            X_train, y_train, X_test, y_test
                        )
                        for emb_dim in emb_dims:
                            model = EEGEmbedder(embedding_dim=emb_dim).to(device)
                            optimizer = optim.Adam(model.parameters(), lr=1e-3)
                            for margin in margins:
                                criterion = nn.TripletMarginWithDistanceLoss(
                                    margin=margin
                                )
                                for epochs in n_epochs_list:
                                    current_iteration += 1
                                    final_loss = train_triplet_online(
                                        model,
                                        train_loader,
                                        criterion,
                                        optimizer,
                                        device,
                                        n_epochs=epochs,
                                    )
                                    embeddings, labels = extract_embeddings_online(
                                        model, test_loader, device
                                    )

                                    print(
                                        f"{current_iteration}/{combinations}, Loss: {final_loss}"
                                    )
                                    result = {
                                        "params": {
                                            "lfreq": lfreq,
                                            "hfreq": hfreq,
                                            "notch_filter": notch_filter,
                                            "tmin": tmin,
                                            "tmax": tmax,
                                            "embedding_dim": emb_dim,
                                            "margin": margin,
                                            "epochs": epochs,
                                            "seed": seed,
                                        },
                                        "final_loss": final_loss,
                                        "embeddings": np.array(embeddings).tolist(),
                                        "labels": np.array(labels).tolist(),
                                    }

                                    save_result_to_json(result, destination_file)
