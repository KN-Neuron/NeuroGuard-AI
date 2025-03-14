import mne

# from load_data import COH_Preprocessing

raw = mne.io.read_raw_fif("data/kolory/Kolory/6d9a8b86@1613.fif", preload=True)
print(raw)
raw.pick_types(eeg=True, stim=False, eog=False, exclude="bads") 
print(raw)