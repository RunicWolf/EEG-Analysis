import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy.stats import kurtosis, skew
from scipy.fft import fft

# Paths
DATA_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\augmented_data.pt"
LABELS_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\augmented_labels.pt"
OUTPUT_DATA_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\enhanced_data.pt"
OUTPUT_LABELS_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\enhanced_labels.pt"

def load_data(data_path, labels_path):
    data = torch.load(data_path).numpy()
    labels = torch.load(labels_path).numpy()
    return data, labels

def create_statistical_features(data):
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    var = np.var(data, axis=1, keepdims=True)
    skewness = skew(data, axis=1).reshape(-1, 1)
    kurt = kurtosis(data, axis=1).reshape(-1, 1)
    return np.hstack([mean, std, var, skewness, kurt])

def create_frequency_features(data):
    fft_features = np.abs(fft(data, axis=1))[:, :data.shape[1] // 2]
    power = np.sum(np.square(fft_features), axis=1, keepdims=True)
    return np.hstack([fft_features, power])

def normalize_features(data, method="zscore"):
    if method == "zscore":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Unsupported normalization method.")
    return scaler.fit_transform(data)

def dimensionality_reduction(data, n_components=0.95):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    return reduced_data

def handle_missing_values(data, method="mean"):
    """
    Handles missing values in the data.

    Args:
        data (np.ndarray): Input data with potential NaN values.
        method (str): Strategy to handle NaN values ("mean", "median", or "drop").

    Returns:
        np.ndarray: Data with missing values handled.
    """
    if method == "mean":
        return np.nan_to_num(data, nan=np.nanmean(data, axis=0))
    elif method == "median":
        return np.nan_to_num(data, nan=np.nanmedian(data, axis=0))
    elif method == "drop":
        return data[~np.isnan(data).any(axis=1)]
    else:
        raise ValueError("Unsupported method for handling missing values.")

if __name__ == "__main__":
    print("Loading augmented data...")
    data, labels = load_data(DATA_PATH, LABELS_PATH)

    print("Creating statistical features...")
    statistical_features = create_statistical_features(data)

    print("Creating frequency features...")
    frequency_features = create_frequency_features(data)

    print("Combining features...")
    combined_features = np.hstack([data, statistical_features, frequency_features])
    print(f"Original feature size: {data.shape[1]}, New feature size: {combined_features.shape[1]}")

    print("Handling missing values...")
    combined_features = handle_missing_values(combined_features, method="mean")

    print("Normalizing features...")
    normalized_features = normalize_features(combined_features, method="zscore")

    print("Applying PCA for dimensionality reduction...")
    reduced_features = dimensionality_reduction(normalized_features, n_components=0.95)

    print("Saving enhanced features...")
    torch.save(torch.tensor(reduced_features, dtype=torch.float32), OUTPUT_DATA_PATH)
    torch.save(torch.tensor(labels, dtype=torch.float32), OUTPUT_LABELS_PATH)
    print(f"Data saved to {OUTPUT_DATA_PATH}, Labels saved to {OUTPUT_LABELS_PATH}.")
