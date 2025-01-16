import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer

# Paths
DATA_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\cleaned_data.pt"
LABELS_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\cleaned_labels.pt"
OUTPUT_DATA_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\transformed_data.pt"
OUTPUT_LABELS_PATH = r"C:\\Users\\prajw\\Desktop\\docs\\All Projects\\EEG\\EEG-Analysis-Chatbot\\processed_data\\Bon_UKB\\transformed_labels.pt"

def load_cleaned_data(data_path, labels_path):
    """
    Load the cleaned dataset and labels.
    """
    data = torch.load(data_path).numpy()
    labels = torch.load(labels_path).numpy()
    return data, labels

def encode_labels(labels):
    """
    Encode categorical labels using one-hot encoding or label encoding.
    """
    unique_classes = len(np.unique(labels))
    
    if unique_classes > 2:
        print("Applying One-Hot Encoding...")
        encoder = OneHotEncoder(sparse_output=False)  # Update sparse to sparse_output
        transformed_labels = encoder.fit_transform(labels.reshape(-1, 1))
    else:
        print("Applying Label Encoding...")
        encoder = LabelEncoder()
        transformed_labels = encoder.fit_transform(labels)
    
    return transformed_labels

def bin_continuous_features(data, n_bins=5, strategy='uniform'):
    """
    Bin continuous features into categories.
    """
    print("Binning continuous features...")
    binner = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    binned_data = binner.fit_transform(data)
    return binned_data

if __name__ == "__main__":
    # Load data
    print("Loading cleaned data...")
    data, labels = load_cleaned_data(DATA_PATH, LABELS_PATH)

    # Encode labels
    print("Encoding labels...")
    transformed_labels = encode_labels(labels)
    print(f"Transformed Labels Shape: {transformed_labels.shape}")

    # Bin features
    print("Binning features...")
    binned_data = bin_continuous_features(data, n_bins=10, strategy='quantile')
    print(f"Binned Data Shape: {binned_data.shape}")

    # Save transformed data
    print("Saving transformed data...")
    torch.save(torch.tensor(binned_data, dtype=torch.float32), OUTPUT_DATA_PATH)
    torch.save(torch.tensor(transformed_labels, dtype=torch.float32), OUTPUT_LABELS_PATH)
    print(f"Transformed data saved to {OUTPUT_DATA_PATH}")
    print(f"Transformed labels saved to {OUTPUT_LABELS_PATH}")
