import os
import mne

dataset = []
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, "datasets", "Kolory")

for filename in os.listdir(data_path):
    if filename.endswith(".fif"):
        im = mne.io.read_raw_fif(os.path.join(data_path, filename), preload=True)
        dataset.append(im)

print(dataset[0].info)
dataset[0].plot