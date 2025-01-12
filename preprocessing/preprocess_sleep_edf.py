import os
import re
import mne
import cupy as cp
from scipy.signal import butter, filtfilt

# Paths
data_dir = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\sleepedf\\physionet.org\\files\\sleep-edfx\\1.0.0"
output_dir = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\processed_data\\Sleep_edf"
os.makedirs(output_dir, exist_ok=True)

# GPU-accelerated bandpass filter
def gpu_bandpass_filter(data, sfreq, l_freq, h_freq):
    nyquist = sfreq / 2.0
    l_norm = l_freq / nyquist
    h_norm = h_freq / nyquist
    b, a = butter(4, [l_norm, h_norm], btype="band")
    data_gpu = cp.array(data)
    filtered_data_gpu = cp.empty_like(data_gpu)
    for i in range(data_gpu.shape[0]):
        filtered_data_gpu[i] = cp.array(filtfilt(b, a, cp.asnumpy(data_gpu[i])))
    return cp.asnumpy(filtered_data_gpu)

# Find matching Hypnogram file
def find_hypnogram_file(psg_file, dataset_dir):
    # Replace the last character of the base name
    base_name = re.sub(r"0-PSG\.edf$", "", psg_file)  # Remove '0-PSG.edf'
    pattern = re.compile(rf"^{base_name}.-Hypnogram\.edf$")  # Regex to match Hypnogram file
    print(f"Looking for Hypnogram matching: {pattern.pattern}")
    for file in os.listdir(dataset_dir):
        print(f"Checking file: {file}")
        if pattern.match(file):
            print(f"Matched file: {file}")
            return os.path.join(dataset_dir, file)
    print(f"No matching Hypnogram file found for: {psg_file}")
    return None

# Process files
def preprocess_sleep_edf(dataset_name, dataset_dir):
    print(f"Processing dataset: {dataset_name}")
    for file_name in os.listdir(dataset_dir):
        if not file_name.endswith("PSG.edf"):
            continue

        psg_path = os.path.join(dataset_dir, file_name)
        hypnogram_path = find_hypnogram_file(file_name, dataset_dir)

        print(f"Processing: {psg_path}")

        try:
            raw_data = mne.io.read_raw_edf(psg_path, preload=True)
            raw_data.set_meas_date(None)
        except Exception as e:
            print(f"Error loading file {psg_path}: {e}")
            continue

        # Apply GPU filtering
        try:
            data = raw_data.get_data()
            sfreq = raw_data.info["sfreq"]
            print("Applying GPU-accelerated filtering...")
            filtered_data = gpu_bandpass_filter(data, sfreq, l_freq=0.5, h_freq=30)
            raw_data._data = filtered_data
        except Exception as e:
            print(f"Error during filtering for file {psg_path}: {e}")
            continue

        # Parse Hypnogram and add annotations
        if hypnogram_path:
            try:
                print(f"Adding annotations from: {hypnogram_path}")
                annotations = mne.read_annotations(hypnogram_path)
                raw_data.set_annotations(annotations)
                print(f"Annotations added: {len(annotations)} events.")
            except Exception as e:
                print(f"Error reading annotations from {hypnogram_path}: {e}")
        else:
            print(f"Hypnogram file not found for: {psg_path}")

        # Save processed file
        output_file = os.path.join(output_dir, f"{dataset_name}_{file_name.replace('PSG.edf', '_raw.fif')}")
        try:
            raw_data.save(output_file, overwrite=True)
            print(f"Processed data saved to: {output_file}")
        except Exception as e:
            print(f"Error saving file {output_file}: {e}")

if __name__ == "__main__":
    preprocess_sleep_edf("sleep-cassette", os.path.join(data_dir, "sleep-cassette"))
    preprocess_sleep_edf("sleep-telemetry", os.path.join(data_dir, "sleep-telemetry"))
