import os
import mne
import cupy as cp
import numpy as np
from scipy.signal import butter, filtfilt

# Paths to required files
data_dir = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\physionet.org\\files\\chbmit\\1.0.0"
output_dir = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Suppress duplicate channel warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Channel names are not unique")

# GPU-accelerated bandpass filter
def gpu_bandpass_filter(data, sfreq, l_freq, h_freq):
    nyquist = sfreq / 2.0
    l_norm = l_freq / nyquist
    h_norm = h_freq / nyquist

    b, a = butter(4, [l_norm, h_norm], btype='band')
    data_gpu = cp.array(data)

    filtered_data_gpu = cp.empty_like(data_gpu)
    for i in range(data_gpu.shape[0]):  # Filter each channel independently
        filtered_data_gpu[i] = cp.array(filtfilt(b, a, cp.asnumpy(data_gpu[i])))

    return cp.asnumpy(filtered_data_gpu)

# Parse summary files
def parse_summary_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    data = {
        "sampling_rate": None,
        "channels": [],
        "edf_files": [],
    }

    current_edf = None

    for line in lines:
        line = line.strip()
        if line.startswith("Data Sampling Rate:"):
            data["sampling_rate"] = int(line.split(":")[1].strip().split()[0])
        elif line.startswith("Channel"):
            parts = line.split(":")
            data["channels"].append(parts[1].strip())
        elif line.startswith("File Name:"):
            current_edf = {
                "file_name": line.split(":")[1].strip(),
                "seizures": []
            }
            data["edf_files"].append(current_edf)
        elif line.startswith("Number of Seizures in File:"):
            seizure_count = int(line.split(":")[1].strip())
            current_edf["seizure_count"] = seizure_count
        elif line.startswith("Seizure Start Time:"):
            start_time = int(line.split(":")[1].strip().split()[0])
            current_edf["seizures"].append({"start_time": start_time})
        elif line.startswith("Seizure End Time:"):
            end_time = int(line.split(":")[1].strip().split()[0])
            current_edf["seizures"][-1]["end_time"] = end_time

    return data

# Preprocess data using summary files
def preprocess_data():
    print(f"Looking for summary files in: {data_dir}")
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        if not os.path.isdir(subdir_path):  # Skip files, process only directories
            continue

        for case in os.listdir(subdir_path):
            if not case.endswith("-summary.txt"):
                continue

            summary_file_path = os.path.join(subdir_path, case)
            print(f"Detected summary file: {summary_file_path}")

            summary_data = parse_summary_file(summary_file_path)
            print(f"Summary File: {summary_file_path}")
            print(f"Sampling Rate: {summary_data['sampling_rate']} Hz")
            print(f"Channels: {summary_data['channels'][:5]}... ({len(summary_data['channels'])} total)")
            print(f"Number of EDF Files: {len(summary_data['edf_files'])}")

            for record in summary_data["edf_files"]:
                edf_path = os.path.join(subdir_path, record["file_name"])
                if not os.path.exists(edf_path):
                    print(f"Missing EDF file: {edf_path}")
                    continue

                print(f"Processing {edf_path}...")

                try:
                    raw_data = mne.io.read_raw_edf(edf_path, preload=True)
                    raw_data.set_meas_date(None)
                except Exception as e:
                    print(f"Error loading file {edf_path}: {e}")
                    continue

                try:
                    data = raw_data.get_data()
                    sfreq = summary_data["sampling_rate"]
                    filtered_data = gpu_bandpass_filter(data, sfreq, l_freq=0.5, h_freq=50)
                    raw_data._data = filtered_data
                except Exception as e:
                    print(f"Error during filtering for file {edf_path}: {e}")
                    continue

                # Add seizure annotations
                try:
                    for seizure in record["seizures"]:
                        raw_data.annotations.append(
                            onset=seizure["start_time"],
                            duration=seizure["end_time"] - seizure["start_time"],
                            description="Seizure"
                        )
                    if record["seizures"]:
                        seizure_times = [
                            (seizure["start_time"], seizure["end_time"]) for seizure in record["seizures"]
                        ]
                        print(f"Seizure times for {record['file_name']}: {seizure_times}")
                    else:
                        print(f"No seizures found for {record['file_name']}.")
                except Exception as e:
                    print(f"Error adding annotations to file {edf_path}: {e}")
                    continue

                output_file = os.path.join(output_dir, record["file_name"].replace(".edf", "_raw.fif"))
                try:
                    raw_data.save(output_file, overwrite=True)
                    print(f"Filtered data saved to: {output_file}")
                except Exception as e:
                    print(f"Error saving file {output_file}: {e}")

# Run preprocessing
if __name__ == "__main__":
    preprocess_data()
