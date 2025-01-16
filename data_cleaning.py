import torch
import numpy as np

def handle_missing_values(data, strategy="mean"):
    """
    Handles missing values in the dataset.

    Args:
        data (torch.Tensor): Input data tensor.
        strategy (str): Strategy to fill missing values - "mean", "median", or "drop".

    Returns:
        torch.Tensor: Data with missing values handled.
    """
    data = data.numpy()  # Convert to NumPy for easier handling

    if strategy == "mean":
        col_mean = np.nanmean(data, axis=0)
        inds = np.where(np.isnan(data))
        data[inds] = np.take(col_mean, inds[1])
    elif strategy == "median":
        col_median = np.nanmedian(data, axis=0)
        inds = np.where(np.isnan(data))
        data[inds] = np.take(col_median, inds[1])
    elif strategy == "drop":
        data = data[~np.isnan(data).any(axis=1)]
    else:
        raise ValueError("Invalid strategy. Choose from 'mean', 'median', or 'drop'.")

    return torch.tensor(data, dtype=torch.float32)

def handle_outliers(data, labels, method="zscore", threshold=3):
    """
    Handles outliers in the dataset.

    Args:
        data (torch.Tensor): Input data tensor.
        labels (torch.Tensor): Corresponding labels tensor.
        method (str): Method for outlier detection - "zscore" or "iqr".
        threshold (float): Threshold for identifying outliers.

    Returns:
        torch.Tensor, torch.Tensor: Data and labels with outliers handled.
    """
    data = data.numpy()  # Convert to NumPy for easier handling
    labels = labels.numpy()

    if method == "zscore":
        z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
        mask = (z_scores < threshold).all(axis=1)
    elif method == "iqr":
        Q1 = np.percentile(data, 25, axis=0)
        Q3 = np.percentile(data, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = np.all((data >= lower_bound) & (data <= upper_bound), axis=1)
    else:
        raise ValueError("Invalid method. Choose from 'zscore' or 'iqr'.")

    data = data[mask]
    labels = labels[mask]  # Filter labels based on the mask

    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

def check_data_consistency(data, expected_range=None):
    """
    Checks data consistency, including ranges and types.

    Args:
        data (torch.Tensor): Input data tensor.
        expected_range (tuple): Tuple of (min, max) values for data.

    Returns:
        torch.Tensor: Consistent data.
    """
    if not torch.is_floating_point(data):
        raise TypeError("Data must be a floating point tensor.")

    if expected_range:
        min_val, max_val = expected_range
        data = torch.clamp(data, min=min_val, max=max_val)

    return data

if __name__ == "__main__":
    # Paths
    data_path = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\original_data.pt"
    labels_path = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\original_labels.pt"

    # Load data and labels
    data = torch.load(data_path)
    labels = torch.load(labels_path)

    print("Handling missing values...")
    data = handle_missing_values(data, strategy="mean")

    print("Handling outliers...")
    data, labels = handle_outliers(data, labels, method="zscore", threshold=3)

    print("Checking data consistency...")
    data = check_data_consistency(data, expected_range=(0, 1))

    # Save cleaned data and labels
    output_data_path = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\cleaned_data.pt"
    output_labels_path = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\cleaned_labels.pt"
    torch.save(data, output_data_path)
    torch.save(labels, output_labels_path)
    print(f"Cleaned data saved to {output_data_path}.")
    print(f"Cleaned labels saved to {output_labels_path}.")
