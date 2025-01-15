import numpy as np
from scipy.stats import zscore

def handle_missing_values(data, method="interpolate"):
    for i in range(data.shape[0]):
        row = data[i]
        if np.any(np.isnan(row)):
            non_nan_indices = np.flatnonzero(~np.isnan(row))
            nan_indices = np.flatnonzero(np.isnan(row))
            row[nan_indices] = np.interp(nan_indices, non_nan_indices, row[non_nan_indices])
    return data

def handle_outliers(data, labels, method="zscore", threshold=3):
    valid_indices = []
    for i, signal in enumerate(data):
        if method == "zscore":
            z_scores = np.abs(zscore(signal))
            if np.all(z_scores < threshold):
                valid_indices.append(i)

    valid_indices = np.array(valid_indices)
    return data[valid_indices], labels[valid_indices]

def normalize_data(data, method="zscore"):
    if method == "zscore":
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif method == "minmax":
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        return (data - data_min) / (data_max - data_min)
