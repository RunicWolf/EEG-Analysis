import os
import numpy as np
import mne
from scipy.signal import butter, filtfilt
import cupy as cp

# Paths
base_dir = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\BCICIV"
calib_dir = os.path.join(base_dir, "BCICIV_1calib_1000Hz_asc")
eval_dir = os.path.join(base_dir, "BCICIV_1eval_1000Hz_asc")
output_dir = os.path.join(base_dir, "processed_data")
os.makedirs(output_dir, exist_ok=True)

# GPU-accelerated bandpass filter
def gpu_bandpass_filter(data, sfreq, l_freq, h_freq):
    nyquist = sfreq / 2.0
    l_norm = l_freq / nyquist
    h_norm = h_freq / nyquist
    b, a = butter(4, [l_norm, h_norm], btype="band")
    data_gpu = cp.array(data)
    filtered_data_gpu = cp.empty_like(data_gpu)
    for i in range(data_gpu.shape[1]):  # Filter each channel independently
        filtered_data_gpu[:, i] = cp.array(filtfilt(b, a, cp.asnumpy(data[:, i])))
    return cp.asnumpy(filtered_data_gpu)

# Parse ASCII files
def parse_ascii_files(cnt_path, nfo_path, mrk_path=None):
    print(f"Loading EEG data from: {cnt_path}")
    eeg_data = np.genfromtxt(cnt_path, delimiter=None)  # Efficient loading for large files
    print(f"EEG data shape: {eeg_data.shape}")

    print(f"Loading metadata from: {nfo_path}")
    with open(nfo_path, "r") as f:
        lines = f.readlines()
        fs = int(lines[0].split(":")[1].strip())  # Extract sampling frequency
        channels = lines[1].split(":")[1].strip().split(",")  # Extract channel names

    # Handle channel mismatch
    if len(channels) != eeg_data.shape[1]:
        print(f"Mismatch: {len(channels)} channels in metadata vs {eeg_data.shape[1]} in EEG data.")
        channels = [f"Ch{i+1}" for i in range(eeg_data.shape[1])]  # Generate generic channel names
    print(f"Sampling frequency: {fs}, Channels: {len(channels)}")

    markers = None
    if mrk_path and os.path.exists(mrk_path):
        print(f"Loading markers from: {mrk_path}")
        markers = np.loadtxt(mrk_path, dtype=float).astype(int)  # Handle mixed types
        print(f"Markers shape: {markers.shape}")

    return eeg_data, fs, channels, markers


# Preprocess EEG data
def preprocess_ascii_data(cnt_path, nfo_path, mrk_path, output_file):
    eeg_data, fs, channels, markers = parse_ascii_files(cnt_path, nfo_path, mrk_path)

    # Apply GPU bandpass filter
    try:
        print("Applying GPU-accelerated bandpass filtering (8–30 Hz)...")
        filtered_data = gpu_bandpass_filter(eeg_data, fs, l_freq=8, h_freq=30)
    except Exception as e:
        print(f"Error during filtering: {e}")
        return

    # Create MNE RawArray
    try:
        info = mne.create_info(ch_names=channels, sfreq=fs, ch_types="eeg")
        print(f"Creating RawArray with data shape {filtered_data.shape} and {len(channels)} channels...")
        raw = mne.io.RawArray(filtered_data.T, info)

        # Add markers as annotations (if available)
        if markers is not None:
            for pos, label in markers:
                onset = pos / fs
                duration = 0  # Markers are instantaneous
                description = "Class1" if label == -1 else "Class2"
                raw.annotations.append(onset, duration, description)
            print(f"Annotations added: {markers.shape[0]} events.")
    except Exception as e:
        print(f"Error creating MNE RawArray: {e}")
        return

    # Save preprocessed data
    try:
        raw.save(output_file, overwrite=True)
        print(f"Processed data saved to: {output_file}")
    except Exception as e:
        print(f"Error saving file {output_file}: {e}")

# Preprocess All Data
def preprocess_ascii_dataset(input_dir, prefix):
    for file_name in os.listdir(input_dir):
        if file_name.endswith("_cnt.txt"):
            base_name = file_name.replace("_cnt.txt", "")
            cnt_path = os.path.join(input_dir, f"{base_name}_cnt.txt")
            nfo_path = os.path.join(input_dir, f"{base_name}_nfo.txt")
            mrk_path = os.path.join(input_dir, f"{base_name}_mrk.txt")  # Only present for calibration data
            output_file = os.path.join(output_dir, f"{prefix}_{base_name}_raw.fif")

            preprocess_ascii_data(cnt_path, nfo_path, mrk_path, output_file)

if __name__ == "__main__":
    print("Processing Calibration Data...")
    preprocess_ascii_dataset(calib_dir, "calib")

    print("\nProcessing Evaluation Data...")
    preprocess_ascii_dataset(eval_dir, "eval")
