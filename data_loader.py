import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, zscore
from scipy.signal import welch
from sklearn.model_selection import train_test_split

# Handle Missing Values
def handle_missing_values(data, method="interpolate"):
    for i in range(data.shape[0]):
        row = data[i]
        if np.any(np.isnan(row)):
            non_nan_indices = np.flatnonzero(~np.isnan(row))
            nan_indices = np.flatnonzero(np.isnan(row))
            row[nan_indices] = np.interp(nan_indices, non_nan_indices, row[non_nan_indices])
    return data

# Handle Outliers
def handle_outliers(data, labels, method="zscore", threshold=3):
    """
    Handles outliers in the EEG data and updates labels accordingly.

    Args:
        data (np.ndarray): EEG data as a NumPy array.
        labels (np.ndarray): Corresponding labels as a NumPy array.
        method (str): Method to handle outliers ('zscore' or 'iqr').
        threshold (float): Threshold for detecting outliers.

    Returns:
        tuple: (data, labels) with outliers removed.
    """
    valid_indices = []
    for i, signal in enumerate(data):
        if method == "zscore":
            z_scores = np.abs(zscore(signal))
            if np.all(z_scores < threshold):  # Keep if no features exceed the threshold
                valid_indices.append(i)
        elif method == "iqr":
            Q1 = np.percentile(signal, 25)
            Q3 = np.percentile(signal, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (threshold * IQR)
            upper_bound = Q3 + (threshold * IQR)
            if np.all((signal >= lower_bound) & (signal <= upper_bound)):
                valid_indices.append(i)
        else:
            raise ValueError(f"Unsupported method for handling outliers: {method}")
    
    valid_indices = np.array(valid_indices)
    return data[valid_indices], labels[valid_indices]



# Normalize Data
def normalize_data(data, method="zscore"):
    if method == "zscore":
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == "minmax":
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        data = (data - data_min) / (data_max - data_min)
    return data

# Extract Features
def extract_features(data, sampling_rate=256):
    features = []
    for signal in data:
        mean = np.mean(signal)
        variance = np.var(signal)
        skewness = skew(signal)
        kurt = kurtosis(signal)
        freqs, psd = welch(signal, fs=sampling_rate)
        psd_mean = np.mean(psd)
        psd_variance = np.var(psd)
        feature_vector = [mean, variance, skewness, kurt, psd_mean, psd_variance]
        features.append(feature_vector)
    return np.array(features)

# Split Data
def split_data(features, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels, test_size=(1 - train_ratio), stratify=labels, random_state=42
    )
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio_adjusted), stratify=y_temp, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

# Visualize Signals
def visualize_signals(data, title="EEG Signals", sample_size=10):
    sample_indices = np.random.choice(len(data), size=sample_size, replace=False)
    plt.figure(figsize=(15, 5))
    for idx in sample_indices:
        plt.plot(data[idx], alpha=0.7)
    plt.title(title)
    plt.xlabel("Time Points")
    plt.ylabel("Amplitude")
    plt.show()

# Main Script
if __name__ == "__main__":
    DATA_DIR = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\Bon"
    OUTPUT_DIR = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading cleaned data...")
    data = torch.load(os.path.join(OUTPUT_DIR, "cleaned_data.pt")).cpu().numpy()
    labels = torch.load(os.path.join(OUTPUT_DIR, "cleaned_labels.pt")).cpu().numpy()

    print("Visualizing original signals...")
    visualize_signals(data, title="Original EEG Signals")

    print("Handling missing values...")
    data = handle_missing_values(data, method="interpolate")

    print("Handling outliers...")
    data, labels = handle_outliers(data, labels, method="zscore", threshold=3)

    print("Normalizing data...")
    data = normalize_data(data, method="zscore")

    print("Extracting features...")
    features = extract_features(data)

    print("Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(features, labels)

    print("Saving split data...")
    torch.save((torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(y_train, dtype=torch.long).to(device)), 
               os.path.join(OUTPUT_DIR, "train.pt"))
    torch.save((torch.tensor(X_val, dtype=torch.float32).to(device), torch.tensor(y_val, dtype=torch.long).to(device)), 
               os.path.join(OUTPUT_DIR, "val.pt"))
    torch.save((torch.tensor(X_test, dtype=torch.float32).to(device), torch.tensor(y_test, dtype=torch.long).to(device)), 
               os.path.join(OUTPUT_DIR, "test.pt"))
    print(f"Data split and saved to {OUTPUT_DIR}.")
