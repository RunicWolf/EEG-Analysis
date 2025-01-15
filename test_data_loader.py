# Test script to verify data_loader.py
from data_loader import load_eeg_data
import torch

if __name__ == "__main__":
    DATA_DIR = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\Bon"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading data...")
    data, labels = load_eeg_data(DATA_DIR, device=device)
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
