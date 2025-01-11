import os
import mne

# Path to the main CHB-MIT dataset directory
data_dir = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\physionet.org\\files\\chbmit\\1.0.0"  # Replace with your path
output_dir = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data"  # Save processed files here

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through all subdirectories (chbxx) and process .edf files
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".edf"):
            file_path = os.path.join(root, file)
            print(f"Processing {file_path}...")

            # Load the raw data
            raw_data = mne.io.read_raw_edf(file_path, preload=True)

            # Fix the `meas_date` issue
            raw_data.set_meas_date(None)

            # Apply bandpass filter (0.5–50 Hz)
            raw_data.filter(l_freq=0.5, h_freq=50)

            # Save the filtered data with a proper filename
            relative_path = os.path.relpath(root, data_dir)
            output_subdir = os.path.join(output_dir, relative_path)
            os.makedirs(output_subdir, exist_ok=True)
            output_file = os.path.join(output_subdir, f"filtered_{file.replace('.edf', '_raw.fif')}")

            raw_data.save(output_file, overwrite=True)
            print(f"Filtered data saved to {output_file}")
