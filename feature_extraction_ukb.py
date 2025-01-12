import os
import torch
import pandas as pd
from mne.io import read_raw_fif
from antropy import sample_entropy

def extract_features(signal, sf):
    """Extract features from a single EEG signal with GPU acceleration."""
    features = {}
    
    # Move signal to GPU
    signal_tensor = torch.tensor(signal, device='cuda', dtype=torch.float32)
    
    # Time-Domain Features
    features['Mean'] = signal_tensor.mean().item()
    features['STD'] = signal_tensor.std().item()
    features['Skewness'] = (torch.mean((signal_tensor - signal_tensor.mean())**3) / signal_tensor.std()**3).item()
    features['Kurtosis'] = (torch.mean((signal_tensor - signal_tensor.mean())**4) / signal_tensor.std()**4 - 3).item()
    features['Peak-to-Peak'] = (signal_tensor.max() - signal_tensor.min()).item()
    features['Zero-Crossing Rate'] = torch.sum(torch.diff(torch.sign(signal_tensor)) != 0).item()
    
    # Frequency-Domain Features (using PyTorch for FFT)
    psd = torch.fft.rfft(signal_tensor)
    psd_power = (psd.real**2 + psd.imag**2).cpu().numpy()  # Convert to NumPy for band calculations
    freqs = torch.fft.rfftfreq(signal_tensor.shape[0], d=1/sf).cpu().numpy()
    bands = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 12),
             'Beta': (12, 30), 'Gamma': (30, 85)}
    for band, (low, high) in bands.items():
        idx = (freqs >= low) & (freqs <= high)
        features[f'{band}_Power'] = psd_power[idx].sum()
    
    # Non-linear Features (Sample Entropy)
    features['Sample_Entropy'] = sample_entropy(signal)
    
    return features

def process_files(base_path, sf=173.61):
    """Process all .fif files and extract features using GPU."""
    all_features = []
    sets = ['F', 'N', 'O', 'S', 'Z']  # Subfolder names for each dataset

    for set_name in sets:
        set_path = os.path.join(base_path, set_name)
        for file_name in os.listdir(set_path):
            if file_name.endswith("_raw.fif"):
                file_path = os.path.join(set_path, file_name)
                print(f"Processing file: {file_path}")
                
                # Load the preprocessed signal
                raw = read_raw_fif(file_path, preload=True, verbose=False)
                data = raw.get_data()  # (channels, samples)
                
                # Extract features for each channel
                for ch_idx, signal in enumerate(data):
                    features = extract_features(signal, sf)
                    features['File'] = file_name
                    features['Set'] = set_name
                    features['Channel'] = raw.ch_names[ch_idx]
                    all_features.append(features)
    
    # Save features to CSV
    features_df = pd.DataFrame(all_features)
    features_df.to_csv("UKB_EEG_extracted_features_cuda.csv", index=False)
    print("Features saved to UKB_EEG_extracted_features_cuda.csv")

# Directory containing preprocessed .fif files
base_path = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\UKB_Bon"

# Run the processing
if torch.cuda.is_available():
    print("Using GPU for feature extraction...")
    process_files(base_path)
else:
    print("GPU not available. Please ensure CUDA is properly configured.")
