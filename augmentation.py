import torch
import numpy as np
import os

# Paths
DATA_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\cleaned_data.pt"
LABELS_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\cleaned_labels.pt"
OUTPUT_DATA_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\augmented_data.pt"
OUTPUT_LABELS_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\augmented_labels.pt"

# Data Augmentation Functions
def time_shift(data, shift=10):
    """Apply time shift to the data."""
    if data.ndim != 2:
        raise ValueError("Input data must be 2-dimensional (samples, features).")
    return np.roll(data, shift, axis=1)

def noise_injection(data, noise_level=0.01):
    """Add random noise to the data."""
    if data.ndim != 2:
        raise ValueError("Input data must be 2-dimensional (samples, features).")
    noise = noise_level * np.random.randn(*data.shape)
    return data + noise

def augment_data(data, labels, augment_factor=1.3):
    """Augment the data and replicate labels."""
    if data.ndim != 2:
        raise ValueError("Input data must be 2-dimensional (samples, features).")
    
    augmented_data = data.copy()
    augmented_labels = labels.copy()
    
    # Apply augmentation with the reduced augment factor
    for _ in range(augment_factor):
        # Apply time shift and noise injection, but don't over-augment
        augmented_data = np.vstack((augmented_data, time_shift(data)))
        augmented_labels = np.hstack((augmented_labels, labels))  # Replicate the labels

        augmented_data = np.vstack((augmented_data, noise_injection(data)))
        augmented_labels = np.hstack((augmented_labels, labels))  # Replicate the labels
    
    # Return augmented data and labels
    return augmented_data, augmented_labels

# Main Script
if __name__ == "__main__":
    print("Loading cleaned data...")
    data = torch.load(DATA_PATH).numpy()
    labels = torch.load(LABELS_PATH).numpy()

    print(f"Original data shape: {data.shape}")
    print(f"Original labels shape: {labels.shape}")

    # Ensure data is 2D
    if data.ndim != 2:
        print(f"Error: Data shape is {data.shape}. Reshaping...")
        data = data.reshape(data.shape[0], -1)

    print("Applying data augmentation...")
    augmented_data, augmented_labels = augment_data(data, labels, augment_factor=1)  # Reduced augment factor

    print(f"Augmented data shape: {augmented_data.shape}")
    print(f"Augmented labels shape: {augmented_labels.shape}")

    # Saving augmented data
    print("Saving augmented data...")
    torch.save(torch.tensor(augmented_data, dtype=torch.float32), OUTPUT_DATA_PATH)
    torch.save(torch.tensor(augmented_labels, dtype=torch.long), OUTPUT_LABELS_PATH)
    print(f"Augmented data saved to {OUTPUT_DATA_PATH}")
    print(f"Augmented labels saved to {OUTPUT_LABELS_PATH}")
