import os
import numpy as np
import mne
from scipy.signal import butter, filtfilt
import cupy as cp

# Paths
base_dir = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\Bon"
sets = {
    "F": os.path.join(base_dir, "f", "F"),
    "N": os.path.join(base_dir, "n", "N"),
    "O": os.path.join(base_dir, "o", "O"),
    "S": os.path.join(base_dir, "s", "S"),
    "Z": os.path.join(base_dir, "z", "Z"),
}
output_dir = os.path.join(base_dir, "processed_data")
os.makedirs(output_dir, exist_ok=True)

# Sampling rate
sfreq = 173.61

# GPU-accelerated bandpass filter
def gpu_bandpass_filter(data, sfreq, l_freq, h_freq):
    nyquist = sfreq / 2.0
    l_norm = l_freq / nyquist
    h_norm = h_freq / nyquist
    b, a = butter(4, [l_norm, h_norm], btype="band")
    data_gpu = cp.array(data)
    filtered_data_gpu = cp.empty_like(data_gpu)
    filtered_data_gpu[:] = cp.array(filtfilt(b, a, cp.asnumpy(data)))
    return cp.asnumpy(filtered_data_gpu)

# Preprocess individual set
def preprocess_set(set_name, set_path):
    print(f"Processing set: {set_name}")
    output_set_dir = os.path.join(output_dir, set_name)
    os.makedirs(output_set_dir, exist_ok=True)

    for file_name in os.listdir(set_path):
        if not file_name.lower().endswith(".txt"):
            continue

        file_path = os.path.join(set_path, file_name)
        print(f"Processing file: {file_path}")

        # Load EEG data
        try:
            eeg_data = np.loadtxt(file_path)
            print(f"Loaded EEG data shape: {eeg_data.shape}")
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            continue

        # Apply GPU-accelerated bandpass filter
        try:
            print("Applying GPU-accelerated bandpass filtering (0.5–40 Hz)...")
            filtered_data = gpu_bandpass_filter(eeg_data, sfreq, l_freq=0.5, h_freq=40)
        except Exception as e:
            print(f"Error during filtering for file {file_path}: {e}")
            continue

        # Create MNE RawArray
        try:
            info = mne.create_info(ch_names=[file_name[:-4]], sfreq=sfreq, ch_types="eeg")
            raw = mne.io.RawArray([filtered_data], info)
        except Exception as e:
            print(f"Error creating RawArray for file {file_path}: {e}")
            continue

        # Save preprocessed data
        output_file = os.path.join(output_set_dir, f"{file_name[:-4]}_raw.fif")
        try:
            raw.save(output_file, overwrite=True)
            print(f"Processed data saved to: {output_file}")
        except Exception as e:
            print(f"Error saving file {output_file}: {e}")

# Main processing function
if __name__ == "__main__":
    for set_name, set_path in sets.items():
        preprocess_set(set_name, set_path)
